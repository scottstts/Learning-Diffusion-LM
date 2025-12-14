use clap::Parser;
use dashmap::DashMap;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use ndarray::Array1;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

/// High-performance, memory-efficient BPE Tokenizer
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input text file
    #[arg(short, long)]
    input: String,

    /// Path to save/load tokenizer JSON
    #[arg(short, long, default_value = "tokenizer_bpe.json")]
    tokenizer: String,

    /// Path to save encoded numpy array
    #[arg(short, long, default_value = "encoded_data.npy")]
    output: String,

    /// Target vocabulary size (excluding post-training special tokens like <MASK>)
    #[arg(short, long, default_value = "8191")]
    vocab_size: usize,

    /// Number of threads (0 = auto)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Encode-only mode: load existing tokenizer and encode input (skip training)
    #[arg(short, long, default_value = "false")]
    encode_only: bool,

    /// Training sample size in MB (0 = use full text). Smaller = less RAM.
    #[arg(long, default_value = "512")]
    sample_mb: usize,

    /// Special token that exists in the text (e.g., "<|endoftext|>"). 
    /// Will be preserved as a single token, not split during training.
    #[arg(long, default_value = "<|endoftext|>")]
    text_separator: String,
}

/// Serializable tokenizer format (compatible with Python version)
#[derive(Serialize, Deserialize)]
struct TokenizerData {
    vocab: FxHashMap<String, String>, // token_id -> token_string
    merges: Vec<(u16, u16)>,
    special_tokens: FxHashMap<String, u16>,
    max_token_len: usize,
}

/// Compact word representation: stores token IDs as u16 with word frequency
#[derive(Clone)]
struct Word {
    tokens: Vec<u16>,
    count: u32, // How many times this word appears in corpus
}

/// High-performance BPE Tokenizer
struct BPETokenizer {
    vocab: Vec<String>,                // token_id -> token_string
    vocab_inv: FxHashMap<String, u16>, // token_string -> token_id
    merges: Vec<(u16, u16)>,
    special_tokens: FxHashMap<String, u16>,
    max_token_len: usize,
}

impl BPETokenizer {
    fn new() -> Self {
        Self {
            vocab: Vec::new(),
            vocab_inv: FxHashMap::default(),
            merges: Vec::new(),
            special_tokens: FxHashMap::default(),
            max_token_len: 1,
        }
    }

    /// Train BPE on the given text using memory-efficient processing
    fn train(&mut self, text: &str, vocab_size: usize, sample_mb: usize, text_separator: &str) {
        let start = Instant::now();

        // Check if text separator exists in text
        let sep_count = text.matches(text_separator).count();
        let has_separator = !text_separator.is_empty() && sep_count > 0;
        if has_separator {
            println!("Found {} occurrences of '{}' in text", sep_count, text_separator);
        }

        // Use sample for training if specified
        let training_text = if sample_mb > 0 && text.len() > sample_mb * 1_000_000 {
            let mut sample_bytes = sample_mb * 1_000_000;
            // Ensure we're at a valid UTF-8 char boundary
            while sample_bytes < text.len() && !text.is_char_boundary(sample_bytes) {
                sample_bytes += 1;
            }
            // Find a clean break point at whitespace
            let end = text[..sample_bytes]
                .rfind(char::is_whitespace)
                .unwrap_or(sample_bytes);
            println!(
                "Using {:.1} MB sample for training (full text: {:.1} MB)",
                end as f64 / 1_000_000.0,
                text.len() as f64 / 1_000_000.0
            );
            &text[..end]
        } else {
            println!("Training on full text ({:.1} MB)", text.len() as f64 / 1_000_000.0);
            text
        };

        // Get unique characters for base vocabulary (from FULL text to handle all chars)
        let chars: FxHashSet<char> = text.chars().collect();
        let mut chars: Vec<char> = chars.into_iter().collect();
        chars.sort();
        println!("Base vocabulary: {} characters", chars.len());

        // Initialize vocabulary with single characters
        self.vocab = chars.iter().map(|c| c.to_string()).collect();
        self.vocab_inv = self
            .vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u16))
            .collect();

        // Add text separator as a special token in the base vocabulary
        // This comes AFTER single chars, so it gets its own ID
        if has_separator {
            let sep_id = self.vocab.len() as u16;
            self.vocab.push(text_separator.to_string());
            self.vocab_inv.insert(text_separator.to_string(), sep_id);
            self.special_tokens.insert(text_separator.to_string(), sep_id);
            println!("Added '{}' as special token with ID {}", text_separator, sep_id);
        }

        // Build word frequency map (memory efficient: deduplicate words)
        println!("Building word frequency map...");
        let word_counts = build_word_counts(training_text, if has_separator { Some(text_separator) } else { None });
        println!("Unique words: {}", word_counts.len());

        // Convert to Word structs with token IDs
        let sep_token = if has_separator { Some(text_separator) } else { None };
        let mut words: Vec<Word> = word_counts
            .into_par_iter()
            .filter_map(|(word_str, count)| {
                // Special token is already a single token, don't split it
                if sep_token == Some(word_str.as_str()) {
                    let token_id = *self.vocab_inv.get(&word_str).unwrap();
                    return Some(Word {
                        tokens: vec![token_id],
                        count: count as u32,
                    });
                }
                
                let tokens: Vec<u16> = word_str
                    .chars()
                    .map(|c| self.vocab_inv.get(&c.to_string()).copied())
                    .collect::<Option<Vec<_>>>()?;
                Some(Word {
                    tokens,
                    count: count as u32,
                })
            })
            .collect();

        let base_vocab_size = self.vocab.len();
        let num_merges = vocab_size.saturating_sub(base_vocab_size);
        println!("Base vocab: {}, performing {} merges...", base_vocab_size, num_merges);

        let pb = ProgressBar::new(num_merges as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        for _ in 0..num_merges {
            // Count pair frequencies (weighted by word count)
            let pair_counts = count_pairs_weighted(&words);

            if pair_counts.is_empty() {
                println!("\nNo more pairs to merge");
                break;
            }

            // Find best pair
            let best_pair = pair_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(pair, _)| pair)
                .unwrap();

            // Create new token
            let new_token_str = format!(
                "{}{}",
                self.vocab[best_pair.0 as usize], 
                self.vocab[best_pair.1 as usize]
            );
            let new_token_id = self.vocab.len() as u16;

            self.vocab.push(new_token_str.clone());
            self.vocab_inv.insert(new_token_str, new_token_id);
            self.merges.push(best_pair);

            // Merge pairs in-place
            merge_pairs_inplace(&mut words, best_pair, new_token_id);

            pb.inc(1);
        }

        pb.finish_with_message("done");

        // Update max token length
        self.max_token_len = self.vocab.iter().map(|t| t.len()).max().unwrap_or(1);

        // Free memory
        drop(words);

        println!(
            "Training complete in {:.2}s! Vocabulary size: {}",
            start.elapsed().as_secs_f64(),
            self.vocab.len()
        );
    }

    /// Add a special token
    fn add_special_token(&mut self, token: &str) -> u16 {
        let token_id = self.vocab.len() as u16;
        self.vocab.push(token.to_string());
        self.vocab_inv.insert(token.to_string(), token_id);
        self.special_tokens.insert(token.to_string(), token_id);
        self.max_token_len = self.max_token_len.max(token.len());
        println!("Added special token '{}' with ID {}", token, token_id);
        token_id
    }

    /// Fast greedy encoding using longest match
    fn encode_chunk(&self, text: &str) -> Vec<u16> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::with_capacity(chars.len() / 3);
        let mut i = 0;

        while i < chars.len() {
            let max_len = self.max_token_len.min(chars.len() - i);

            let mut found = false;
            for length in (1..=max_len).rev() {
                let substr: String = chars[i..i + length].iter().collect();
                if let Some(&token_id) = self.vocab_inv.get(&substr) {
                    tokens.push(token_id);
                    i += length;
                    found = true;
                    break;
                }
            }

            if !found {
                panic!("Unknown character at position {}: {:?}", i, chars[i]);
            }
        }

        tokens
    }

    /// Streaming parallel encoding - processes in chunks to limit memory
    fn encode_streaming(&self, text: &str, output_path: &str) -> std::io::Result<usize> {
        let start = Instant::now();
        println!("Encoding {} characters...", text.len());

        // Split into chunks
        let chunk_size = 4 * 1024 * 1024; // 4MB chunks
        let chunks = split_text_chunks(text, chunk_size);
        println!("Processing {} chunks...", chunks.len());

        let pb = ProgressBar::new(chunks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        // Encode all chunks in parallel
        let encoded_chunks: Vec<Vec<u16>> = chunks
            .par_iter()
            .map(|chunk| self.encode_chunk(chunk))
            .collect();

        pb.finish();

        // Calculate total size
        let total_tokens: usize = encoded_chunks.iter().map(|c| c.len()).sum();
        
        // Write directly to npy format
        println!("Writing {} tokens to {}...", total_tokens, output_path);
        write_npy_streaming(&encoded_chunks, output_path)?;

        let file_size = std::fs::metadata(output_path)?.len();
        println!(
            "Encoding complete in {:.2}s! {} tokens ({:.1} MB, compression: {:.2}x)",
            start.elapsed().as_secs_f64(),
            total_tokens,
            file_size as f64 / 1_000_000.0,
            text.len() as f64 / total_tokens as f64
        );

        Ok(total_tokens)
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: &[u16]) -> String {
        tokens
            .iter()
            .map(|&id| self.vocab[id as usize].as_str())
            .collect()
    }

    /// Save tokenizer to JSON (Python-compatible format)
    fn save(&self, path: &str) -> std::io::Result<()> {
        let data = TokenizerData {
            vocab: self
                .vocab
                .iter()
                .enumerate()
                .map(|(i, s)| (i.to_string(), s.clone()))
                .collect(),
            merges: self.merges.clone(),
            special_tokens: self.special_tokens.clone(),
            max_token_len: self.max_token_len,
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data)?;
        println!("Tokenizer saved to {}", path);
        Ok(())
    }

    /// Load tokenizer from JSON file
    fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let data: TokenizerData = serde_json::from_reader(file)?;

        // Reconstruct vocab vector from hashmap
        let mut vocab: Vec<(usize, String)> = data
            .vocab
            .into_iter()
            .map(|(k, v)| (k.parse::<usize>().unwrap(), v))
            .collect();
        vocab.sort_by_key(|(k, _)| *k);
        let vocab: Vec<String> = vocab.into_iter().map(|(_, v)| v).collect();

        // Build inverse vocab
        let vocab_inv: FxHashMap<String, u16> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u16))
            .collect();

        let max_token_len = data
            .max_token_len
            .max(vocab.iter().map(|t| t.len()).max().unwrap_or(1));

        println!(
            "Tokenizer loaded from {} (vocab size: {})",
            path,
            vocab.len()
        );

        Ok(Self {
            vocab,
            vocab_inv,
            merges: data.merges,
            special_tokens: data.special_tokens,
            max_token_len,
        })
    }
}

/// Build word frequency map from text, treating separator as its own word
fn build_word_counts(text: &str, separator: Option<&str>) -> FxHashMap<String, usize> {
    let counts: DashMap<String, usize, FxBuildHasher> =
        DashMap::with_hasher(FxBuildHasher);

    match separator {
        Some(sep) => {
            // Split by separator first, then by whitespace
            for part in text.split(sep) {
                // Count the separator itself (except we get n-1 separators for n parts)
                // We'll add separator count separately
                
                // Process the part
                let chunk = part;
                let mut start = 0;
                let mut in_whitespace = chunk.chars().next().map_or(false, |c| c.is_whitespace());

                for (i, c) in chunk.char_indices() {
                    let is_ws = c.is_whitespace();
                    if is_ws != in_whitespace {
                        if i > start {
                            let word = &chunk[start..i];
                            *counts.entry(word.to_string()).or_insert(0) += 1;
                        }
                        start = i;
                        in_whitespace = is_ws;
                    }
                }
                if start < chunk.len() {
                    let word = &chunk[start..];
                    *counts.entry(word.to_string()).or_insert(0) += 1;
                }
            }
            
            // Count separator occurrences
            let sep_count = text.matches(sep).count();
            if sep_count > 0 {
                counts.insert(sep.to_string(), sep_count);
            }
        }
        None => {
            // Original logic: process in parallel chunks
            let chunk_size = 1_000_000;
            let chunks = split_text_chunks(text, chunk_size);

            chunks.par_iter().for_each(|chunk| {
                let mut start = 0;
                let mut in_whitespace = chunk.chars().next().map_or(false, |c| c.is_whitespace());

                for (i, c) in chunk.char_indices() {
                    let is_ws = c.is_whitespace();
                    if is_ws != in_whitespace {
                        if i > start {
                            let word = &chunk[start..i];
                            *counts.entry(word.to_string()).or_insert(0) += 1;
                        }
                        start = i;
                        in_whitespace = is_ws;
                    }
                }
                if start < chunk.len() {
                    let word = &chunk[start..];
                    *counts.entry(word.to_string()).or_insert(0) += 1;
                }
            });
        }
    }

    counts.into_iter().collect()
}

/// Split text into chunks at word boundaries (UTF-8 safe)
fn split_text_chunks(text: &str, target_size: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let mut end = (start + target_size).min(text.len());
        
        // Ensure we're at a valid UTF-8 char boundary
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }

        // Find a good split point (whitespace)
        let actual_end = if end < text.len() {
            // Search backwards for whitespace
            let search_region = &text[start..end];
            if let Some(pos) = search_region.rfind(|c: char| c.is_whitespace()) {
                let boundary = start + pos + 1;
                // Make sure this is also a valid char boundary
                if text.is_char_boundary(boundary) {
                    boundary
                } else {
                    end
                }
            } else {
                end
            }
        } else {
            end
        };

        if actual_end > start {
            chunks.push(&text[start..actual_end]);
        }
        start = actual_end;
    }

    chunks
}

/// Count pair frequencies weighted by word count
fn count_pairs_weighted(words: &[Word]) -> FxHashMap<(u16, u16), u64> {
    let counts: DashMap<(u16, u16), u64, FxBuildHasher> =
        DashMap::with_hasher(FxBuildHasher);

    words.par_iter().for_each(|word| {
        if word.tokens.len() >= 2 {
            for window in word.tokens.windows(2) {
                let pair = (window[0], window[1]);
                *counts.entry(pair).or_insert(0) += word.count as u64;
            }
        }
    });

    counts.into_iter().collect()
}

/// Merge pairs in-place (modifies words directly)
fn merge_pairs_inplace(words: &mut [Word], pair: (u16, u16), new_token: u16) {
    words.par_iter_mut().for_each(|word| {
        if word.tokens.len() < 2 {
            return;
        }

        let mut i = 0;
        let mut write = 0;

        while i < word.tokens.len() {
            if i + 1 < word.tokens.len()
                && word.tokens[i] == pair.0
                && word.tokens[i + 1] == pair.1
            {
                word.tokens[write] = new_token;
                write += 1;
                i += 2;
            } else {
                word.tokens[write] = word.tokens[i];
                write += 1;
                i += 1;
            }
        }

        word.tokens.truncate(write);
    });
}

/// Write chunks directly to numpy format (avoids creating one huge array)
fn write_npy_streaming(chunks: &[Vec<u16>], path: &str) -> std::io::Result<()> {
    let total_len: usize = chunks.iter().map(|c| c.len()).sum();

    // For smaller outputs, use ndarray-npy directly
    if total_len < 100_000_000 {
        // < 100M tokens
        let mut all_tokens = Vec::with_capacity(total_len);
        for chunk in chunks {
            all_tokens.extend_from_slice(chunk);
        }
        let array = Array1::from(all_tokens);
        ndarray_npy::write_npy(path, &array).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        return Ok(());
    }

    // For very large outputs, write raw npy manually
    let mut file = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);

    // NPY v1.0 header
    let header = format!(
        "{{'descr': '<u2', 'fortran_order': False, 'shape': ({},), }}",
        total_len
    );
    let header_len = header.len();
    let padding = (64 - ((10 + header_len) % 64)) % 64;
    let padded_header = format!("{}{}\n", header, " ".repeat(padding));

    // Magic number + version
    file.write_all(&[0x93u8])?;
    file.write_all(b"NUMPY")?;
    file.write_all(&[0x01, 0x00])?; // Version 1.0

    // Header length (little-endian u16)
    let header_total_len = padded_header.len() as u16;
    file.write_all(&header_total_len.to_le_bytes())?;

    // Header
    file.write_all(padded_header.as_bytes())?;

    // Data (little-endian u16 values)
    for chunk in chunks {
        for &token in chunk {
            file.write_all(&token.to_le_bytes())?;
        }
    }

    file.flush()?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // Set thread count
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

    println!("=== High-Performance BPE Tokenizer ===");
    println!("Using {} threads", rayon::current_num_threads());

    // Memory-map the input file
    let file = File::open(&args.input)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let text = std::str::from_utf8(&mmap).expect("Input file must be valid UTF-8");
    println!(
        "Loaded {} ({:.1} MB)",
        args.input,
        text.len() as f64 / 1_000_000.0
    );

    let tokenizer = if args.encode_only {
        println!(
            "\n[Encode-only mode] Loading tokenizer from {}...",
            args.tokenizer
        );
        BPETokenizer::load(&args.tokenizer)?
    } else {
        let mut tokenizer = BPETokenizer::new();
        tokenizer.train(text, args.vocab_size, args.sample_mb, &args.text_separator);

        // Add MASK token (the only post-training special token)
        tokenizer.add_special_token("<MASK>");

        println!("Final vocabulary size: {}", tokenizer.vocab.len());
        tokenizer.save(&args.tokenizer)?;
        tokenizer
    };

    // Encode and save (streaming)
    let _total_tokens = tokenizer.encode_streaming(text, &args.output)?;

    // Verify with a small sample
    println!("\nVerification:");
    let sample_end = text.char_indices().nth(2000).map(|(i, _)| i).unwrap_or(text.len());
    let sample_text = &text[..sample_end];
    let sample_tokens = tokenizer.encode_chunk(sample_text);
    let decoded = tokenizer.decode(&sample_tokens[..100.min(sample_tokens.len())]);
    let display_end = decoded.char_indices().nth(200).map(|(i, _)| i).unwrap_or(decoded.len());
    println!("First 100 tokens decode to: {}...", &decoded[..display_end]);

    // Show special tokens
    println!("\nSpecial tokens:");
    for (token, id) in &tokenizer.special_tokens {
        println!("  '{}' -> {}", token, id);
    }

    println!("\n=== Done! ===");
    Ok(())
}
