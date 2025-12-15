import React, { useState, useEffect, useRef, useCallback } from 'react';

// Configuration constants matching Python
const MASK_COLOR = '#b482c8';
const TEXT_COLOR = '#dcdce6';
const NEWLY_UNMASKED_COLOR = '#64dc96';
const NEWLY_UNMASKED_BG = '#1e3c2d';
const GRID_COLOR = '#232630';
const CELL_BG_COLOR = '#2e3240';

export default function MDLMVisualizer() {
    const [history, setHistory] = useState(null);
    const [vocab, setVocab] = useState({});
    const [maskTokenId, setMaskTokenId] = useState(null);
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [hasPlaybackFinished, setHasPlaybackFinished] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [temperature, setTemperature] = useState(0.7);
    const [stepCountConfig] = useState(100);
    const [error, setError] = useState(null);

    // Prompt mode state
    const [usePrompt, setUsePrompt] = useState(false);
    const [seqLen, setSeqLen] = useState(1024);
    const [endoftextToken, setEndoftextToken] = useState(242);
    const [promptSegments, setPromptSegments] = useState([]);
    const [tokenizingId, setTokenizingId] = useState(null);
    const [promptPositions, setPromptPositions] = useState(new Set()); // Track which positions are user prompts
    const [promptWordBoundaries, setPromptWordBoundaries] = useState(new Set()); // Track word start positions in prompts

    const animationFrameRef = useRef();
    const lastFrameTimeRef = useRef(0);
    const tokenizeTimeoutRef = useRef({});
    const FPS = 10;
    const frameInterval = 1000 / FPS;

    // Fetch info on mount
    useEffect(() => {
        fetch('/api/info')
            .then(res => res.json())
            .then(data => {
                setSeqLen(data.seq_len || 1024);
                setMaskTokenId(data.mask_token);
                setEndoftextToken(data.endoftext_token);
            })
            .catch(err => console.error('Failed to fetch info:', err));
    }, []);

    const normalizeTokenString = (raw) => {
        return (raw || '').replace(/Ġ/g, ' ').replace(/\r?\n/g, '');
    };

    // Handle Animation Loop
    useEffect(() => {
        if (isPlaying && history && currentStep < history.length - 1) {
            const animate = (time) => {
                if (time - lastFrameTimeRef.current > frameInterval) {
                    setCurrentStep(prev => {
                        const next = prev + 1;
                        if (next >= history.length - 1) {
                            setIsPlaying(false);
                            setHasPlaybackFinished(true);
                            return history.length - 1;
                        }
                        return next;
                    });
                    lastFrameTimeRef.current = time;
                }
                animationFrameRef.current = requestAnimationFrame(animate);
            };
            animationFrameRef.current = requestAnimationFrame(animate);
        }
        return () => cancelAnimationFrame(animationFrameRef.current);
    }, [isPlaying, history, currentStep, frameInterval]);

    // Tokenize text for a segment (debounced)
    const tokenizeSegment = useCallback(async (segmentId, text) => {
        if (!text.trim()) {
            setPromptSegments(prev => prev.map(s =>
                s.id === segmentId ? { ...s, tokens: [], tokenIds: [], wordTokenBoundaries: [0] } : s
            ));
            return;
        }

        setTokenizingId(segmentId);
        try {
            const response = await fetch('/api/tokenize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            setPromptSegments(prev => prev.map(s =>
                s.id === segmentId ? {
                    ...s,
                    tokens: data.tokens || [],
                    tokenIds: data.token_ids || [],
                    wordTokenBoundaries: data.word_boundaries || [0]
                } : s
            ));
        } catch (err) {
            console.error('Tokenization failed:', err);
        } finally {
            setTokenizingId(null);
        }
    }, []);

    // Debounced tokenization
    const handleTextChange = (segmentId, text) => {
        setPromptSegments(prev => prev.map(s =>
            s.id === segmentId ? { ...s, text } : s
        ));

        // Clear existing timeout for this segment
        if (tokenizeTimeoutRef.current[segmentId]) {
            clearTimeout(tokenizeTimeoutRef.current[segmentId]);
        }

        // Set new timeout
        tokenizeTimeoutRef.current[segmentId] = setTimeout(() => {
            tokenizeSegment(segmentId, text);
        }, 300);
    };

    const addPromptSegment = () => {
        const newId = Date.now();
        setPromptSegments(prev => [...prev, {
            id: newId,
            position: 1, // After start token
            text: '',
            tokens: [],
            tokenIds: [],
            wordTokenBoundaries: [0]
        }]);
    };

    const removePromptSegment = (id) => {
        setPromptSegments(prev => prev.filter(s => s.id !== id));
    };

    const updateSegmentPosition = (id, position) => {
        const pos = Math.max(1, Math.min(seqLen - 2, parseInt(position) || 1));
        setPromptSegments(prev => prev.map(s =>
            s.id === id ? { ...s, position: pos } : s
        ));
    };

    // Build prompt array from segments and track prompt positions + word boundaries
    const buildPromptArray = useCallback(() => {
        if (!usePrompt || promptSegments.length === 0) {
            return { prompt: null, positions: new Set(), wordBoundaries: new Set() };
        }

        // Ensure we have valid token IDs
        if (maskTokenId === null || endoftextToken === null) {
            console.error('Token IDs not loaded yet');
            return { prompt: null, positions: new Set(), wordBoundaries: new Set() };
        }

        // Initialize with all mask tokens
        const prompt = new Array(seqLen).fill(maskTokenId);
        const positions = new Set();
        const wordBoundaries = new Set();

        // Set start and end tokens
        prompt[0] = endoftextToken;
        prompt[seqLen - 1] = endoftextToken;

        // Place each segment's tokens at their position
        for (const segment of promptSegments) {
            const startPos = segment.position;
            const tokenIds = segment.tokenIds || [];

            // Place tokens
            for (let i = 0; i < tokenIds.length; i++) {
                const pos = startPos + i;
                if (pos > 0 && pos < seqLen - 1) {
                    prompt[pos] = tokenIds[i];
                    positions.add(pos);
                }
            }

            // Use wordTokenBoundaries calculated during tokenization
            const boundaries = segment.wordTokenBoundaries || [0];
            for (const boundaryIndex of boundaries) {
                const pos = startPos + boundaryIndex;
                if (pos > 0 && pos < seqLen - 1) {
                    wordBoundaries.add(pos);
                }
            }
        }

        return { prompt, positions, wordBoundaries };
    }, [usePrompt, promptSegments, seqLen, maskTokenId, endoftextToken]);

    // Calculate which positions are occupied by prompts
    const getOccupiedPositions = useCallback(() => {
        const occupied = new Set([0, seqLen - 1]); // Start and end always occupied
        for (const segment of promptSegments) {
            const startPos = segment.position;
            const tokenIds = segment.tokenIds || [];
            for (let i = 0; i < tokenIds.length; i++) {
                const pos = startPos + i;
                if (pos > 0 && pos < seqLen - 1) {
                    occupied.add(pos);
                }
            }
        }
        return occupied;
    }, [promptSegments, seqLen]);

    // Check for overlapping segments - returns set of segment IDs that overlap
    const getOverlappingSegments = useCallback(() => {
        const overlapping = new Set();
        const positionOwners = new Map(); // position -> first segment ID that claims it

        for (const segment of promptSegments) {
            const startPos = segment.position;
            const tokenCount = segment.tokenIds?.length || 0;

            for (let i = 0; i < tokenCount; i++) {
                const pos = startPos + i;
                if (pos > 0 && pos < seqLen - 1) {
                    if (positionOwners.has(pos)) {
                        // This position is already claimed - both segments overlap
                        overlapping.add(positionOwners.get(pos));
                        overlapping.add(segment.id);
                    } else {
                        positionOwners.set(pos, segment.id);
                    }
                }
            }
        }
        return overlapping;
    }, [promptSegments, seqLen]);

    const overlappingSegments = getOverlappingSegments();
    const hasOverlap = overlappingSegments.size > 0;

    const handleGenerate = async () => {
        setIsLoading(true);
        setError(null);
        setHistory(null);
        setCurrentStep(0);
        setIsPlaying(false);
        setHasPlaybackFinished(false);

        try {
            const { prompt: promptArray, positions, wordBoundaries } = buildPromptArray();
            setPromptPositions(positions); // Store which positions are user prompts
            setPromptWordBoundaries(wordBoundaries); // Store word boundary positions

            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    temperature: temperature,
                    steps: stepCountConfig,
                    prompt: promptArray
                })
            });

            if (!response.ok) {
                const errJson = await response.json();
                throw new Error(errJson.error || 'Request failed');
            }

            const data = await response.json();
            setHistory(data.history);
            setVocab(data.vocab);
            setMaskTokenId(data.mask_token);

            if (data.history && data.history.length > 1) {
                setIsPlaying(true);
            } else {
                setHasPlaybackFinished(true);
            }
        } catch (err) {
            console.error(err);
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const getTokenDisplay = (tokenId) => {
        if (tokenId === maskTokenId) return '?';
        const raw = vocab[tokenId] || '';
        return normalizeTokenString(raw);
    };

    // Helper to merge tokens into word groups for display
    const getDisplayGroups = (tokens) => {
        if (!tokens) return [];

        const groups = [];
        let currentGroup = [];
        let currentGroupIsPrompt = false;

        tokens.forEach((id, position) => {
            const rawToken = vocab[id] || '';
            const isMask = id === maskTokenId;
            const isEndOfText = rawToken === '<|endoftext|>';
            const isPrompt = promptPositions.has(position);
            const newlineCount = (rawToken.match(/\r?\n/g) || []).length;

            if (isEndOfText) {
                if (currentGroup.length > 0) {
                    groups.push({ tokens: currentGroup, isPrompt: currentGroupIsPrompt });
                    currentGroup = [];
                }
                groups.push({ tokens: [{ id, position, text: '', isMask: false, isEndOfText: true, isPrompt: false }], isPrompt: false });
                return;
            }

            if (isMask) {
                if (currentGroup.length > 0) {
                    groups.push({ tokens: currentGroup, isPrompt: currentGroupIsPrompt });
                    currentGroup = [];
                }
                groups.push({ tokens: [{ id, position, text: '?', isMask: true, isEndOfText: false, isBreak: false, isPrompt: false }], isPrompt: false });
                return;
            }

            if (newlineCount > 0) {
                if (currentGroup.length > 0) {
                    groups.push({ tokens: currentGroup, isPrompt: currentGroupIsPrompt });
                    currentGroup = [];
                }
                for (let i = 0; i < newlineCount; i++) {
                    groups.push({ tokens: [{ id, position, text: '', isMask: false, isEndOfText: false, isBreak: true, isPrompt }], isPrompt });
                }
                const remainderRaw = rawToken.replace(/\r?\n/g, '');
                const remainderText = normalizeTokenString(remainderRaw);
                if (remainderText.trim() !== '') {
                    currentGroup.push({ id, position, text: remainderText, isMask: false, isEndOfText: false, isBreak: false, isPrompt });
                    currentGroupIsPrompt = isPrompt;
                }
                return;
            }

            const isNewWord = rawToken.startsWith(' ') ||
                rawToken.startsWith('Ġ') ||
                rawToken.startsWith('"') ||
                rawToken.startsWith('"');

            // Also break group when transitioning between prompt and non-prompt tokens
            const promptTransition = currentGroup.length > 0 && currentGroupIsPrompt !== isPrompt;

            // For prompt tokens, use word boundaries from original text
            const isPromptWordBoundary = isPrompt && promptWordBoundaries.has(position);

            if ((isNewWord || promptTransition || isPromptWordBoundary) && currentGroup.length > 0) {
                groups.push({ tokens: currentGroup, isPrompt: currentGroupIsPrompt });
                currentGroup = [];
            }

            const tokenDisplay = getTokenDisplay(id);
            if (tokenDisplay.trim() === '') {
                if (currentGroup.length > 0) {
                    groups.push({ tokens: currentGroup, isPrompt: currentGroupIsPrompt });
                    currentGroup = [];
                }
                return;
            }

            currentGroup.push({ id, position, text: tokenDisplay, isMask, isEndOfText, isBreak: false, isPrompt });
            currentGroupIsPrompt = isPrompt;
        });
        if (currentGroup.length > 0) {
            groups.push({ tokens: currentGroup, isPrompt: currentGroupIsPrompt });
        }

        return groups;
    };

    const currentTokens = history ? history[currentStep] : [];
    const groups = getDisplayGroups(currentTokens);
    const lastStep = history ? Math.max(0, history.length - 1) : 0;
    const showPlaybackControls = Boolean(history) && !hasPlaybackFinished;
    const showTimeScrubber = Boolean(history) && hasPlaybackFinished;
    const occupiedPositions = getOccupiedPositions();
    const occupiedCount = occupiedPositions.size - 2; // Exclude start/end

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            backgroundColor: '#1a1b26',
            color: '#a9b1d6',
            overflow: 'hidden',
            paddingBottom: '20px'
        }}>
            {/* Header */}
            <div style={{
                padding: '20px 40px',
                borderBottom: '1px solid #232433',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: '#16161e'
            }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: '1.5rem', color: '#7aa2f7' }}>MDLM Denoising Visualization</h1>
                    {history && (
                        <div style={{ fontSize: '0.9rem', color: '#565f89', marginTop: '5px' }}>
                            Step {currentStep} / {history.length - 1}
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                    {showTimeScrubber && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', minWidth: '260px' }}>
                            <label style={{ fontSize: '0.8rem' }}>
                                Time: {currentStep} / {lastStep}
                            </label>
                            <input
                                type="range"
                                min="0"
                                max={lastStep}
                                step="1"
                                value={currentStep}
                                onChange={(e) => {
                                    setIsPlaying(false);
                                    setCurrentStep(parseInt(e.target.value, 10));
                                }}
                                disabled={isLoading || !history}
                            />
                        </div>
                    )}

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                        <label style={{ fontSize: '0.8rem' }}>Temperature: {temperature}</label>
                        <input
                            type="range"
                            min="0.1"
                            max="2.0"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            disabled={isLoading || (Boolean(history) && !hasPlaybackFinished)}
                        />
                    </div>

                    {showPlaybackControls && (
                        <button
                            onClick={isPlaying ? () => setIsPlaying(false) : () => setIsPlaying(true)}
                            disabled={isLoading}
                            style={{
                                width: '90px',
                                display: 'inline-flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                                whiteSpace: 'nowrap'
                            }}
                        >
                            {isPlaying ? 'Pause' : 'Play'}
                        </button>
                    )}

                    <button
                        onClick={handleGenerate}
                        disabled={(isLoading && !history) || hasOverlap}
                        title={hasOverlap ? 'Fix overlapping prompt segments first' : undefined}
                        style={{
                            backgroundColor: hasOverlap ? '#565f89' : '#7aa2f7',
                            color: '#1a1b26',
                            fontWeight: 'bold',
                            opacity: isLoading ? 0.7 : 1,
                            width: '160px',
                            display: 'inline-flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            whiteSpace: 'nowrap',
                            cursor: hasOverlap ? 'not-allowed' : 'pointer'
                        }}
                    >
                        {isLoading ? 'Running...' : 'Start'}
                    </button>
                </div>
            </div>

            {/* Main Content Area - Split View */}
            <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
                {/* Left: Token Grid */}
                <div style={{
                    flex: 1,
                    minHeight: 0,
                    overflowY: 'auto',
                    padding: '30px 40px',
                    paddingBottom: '100px',
                    display: 'flex',
                    flexWrap: 'wrap',
                    alignContent: 'flex-start',
                    gap: '3px'
                }}>
                    {error && (
                        <div style={{
                            width: '100%',
                            padding: '20px',
                            backgroundColor: '#3b2426',
                            color: '#f7768e',
                            borderRadius: '8px'
                        }}>
                            Error: {error}
                        </div>
                    )}

                    {!history && !isLoading && !error && (
                        <div style={{ width: '100%', textAlign: 'center', marginTop: '100px', opacity: 0.5 }}>
                            Press "Start" to start generation.
                        </div>
                    )}

                    {groups.map((group, gIdx) => {
                        const groupTokens = group.tokens;
                        const isPromptGroup = group.isPrompt;
                        const isBreakGroup = groupTokens.length === 1 && (groupTokens[0].isEndOfText || groupTokens[0].isBreak);
                        if (isBreakGroup) {
                            return (
                                <div
                                    key={gIdx}
                                    style={{ flexBasis: '100%', width: '100%', height: '20px' }}
                                />
                            );
                        }

                        const mergedTextRaw = groupTokens.map((t) => (t.isMask ? '?' : t.text)).join('');
                        const mergedText = mergedTextRaw.replace(/^\s+/, '');
                        if (mergedText.length === 0) return null;

                        let bg = CELL_BG_COLOR;
                        let color = TEXT_COLOR;
                        let border = '1px solid transparent';

                        const isMaskGroup = groupTokens.length === 1 && groupTokens[0].isMask;
                        if (isMaskGroup) {
                            bg = '#2a2635';
                            color = MASK_COLOR;
                            border = '1px solid #45354a';
                        } else if (isPromptGroup) {
                            // Highlight user prompt tokens
                            bg = '#1e3a5f';
                            color = '#7dcfff';
                            border = '1px solid #3d5a80';
                        }

                        return (
                            <div
                                key={gIdx}
                                title={isPromptGroup ? 'User prompt' : undefined}
                                style={{
                                    padding: '2px 6px',
                                    minWidth: isMaskGroup ? '14px' : 'auto',
                                    textAlign: 'center',
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    backgroundColor: bg,
                                    color: color,
                                    fontFamily: 'monospace',
                                    fontSize: '14px',
                                    border: border,
                                    borderRadius: '4px',
                                    transition: 'background-color 0.2s',
                                    whiteSpace: 'pre'
                                }}
                            >
                                {mergedText}
                            </div>
                        );
                    })}
                </div>

                {/* Right: Prompt Panel */}
                <div style={{
                    width: '550px',
                    minWidth: '550px',
                    borderLeft: '1px solid #232433',
                    backgroundColor: '#16161e',
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden'
                }}>
                    {/* Panel Header */}
                    <div style={{
                        padding: '20px 25px',
                        borderBottom: '1px solid #232433',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <span style={{ fontWeight: 'bold', color: '#7aa2f7' }}>Use Prompt</span>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                            <span style={{ fontSize: '0.85rem', color: usePrompt ? '#7aa2f7' : '#565f89' }}>
                                {usePrompt ? 'ON' : 'OFF'}
                            </span>
                            <div
                                onClick={() => setUsePrompt(!usePrompt)}
                                style={{
                                    width: '44px',
                                    height: '24px',
                                    backgroundColor: usePrompt ? '#7aa2f7' : '#3b3d4d',
                                    borderRadius: '12px',
                                    position: 'relative',
                                    transition: 'background-color 0.2s',
                                    cursor: 'pointer'
                                }}
                            >
                                <div style={{
                                    width: '18px',
                                    height: '18px',
                                    backgroundColor: '#fff',
                                    borderRadius: '50%',
                                    position: 'absolute',
                                    top: '3px',
                                    left: usePrompt ? '23px' : '3px',
                                    transition: 'left 0.2s'
                                }} />
                            </div>
                        </label>
                    </div>

                    {/* Panel Content */}
                    <div style={{
                        flex: 1,
                        overflowY: 'auto',
                        padding: '20px 25px',
                        paddingBottom: '100px',
                        opacity: usePrompt ? 1 : 0.4,
                        pointerEvents: usePrompt ? 'auto' : 'none',
                        transition: 'opacity 0.2s'
                    }}>
                        {!usePrompt ? (
                            <div style={{ textAlign: 'center', color: '#565f89', marginTop: '40px' }}>
                                <p>Prompt mode is disabled.</p>
                                <p style={{ fontSize: '0.85rem' }}>
                                    Generation will start with all mask tokens
                                    (except start/end tokens).
                                </p>
                            </div>
                        ) : (
                            <>
                                <p style={{ fontSize: '0.85rem', color: '#565f89', marginBottom: '15px' }}>
                                    Add text at any position in the sequence. Positions not filled will be mask tokens.
                                </p>

                                {/* Stats */}
                                <div style={{
                                    display: 'flex',
                                    gap: '15px',
                                    marginBottom: '15px',
                                    fontSize: '0.8rem',
                                    color: '#7aa2f7'
                                }}>
                                    <span>Sequence: {seqLen}</span>
                                    <span>Filled: {occupiedCount}</span>
                                    <span>Masks: {seqLen - 2 - occupiedCount}</span>
                                </div>

                                {/* Prompt Segments */}
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    {promptSegments.map((segment, idx) => {
                                        const isOverlapping = overlappingSegments.has(segment.id);
                                        return (
                                        <div
                                            key={segment.id}
                                            style={{
                                                backgroundColor: isOverlapping ? '#2d1a1a' : '#1a1b26',
                                                border: isOverlapping ? '1px solid #f7768e' : '1px solid #2d2f3d',
                                                borderRadius: '8px',
                                                padding: '12px'
                                            }}
                                        >
                                            <div style={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                                marginBottom: '10px'
                                            }}>
                                                <span style={{ fontSize: '0.8rem', color: '#7aa2f7' }}>
                                                    Segment {idx + 1}
                                                </span>
                                                <button
                                                    onClick={() => removePromptSegment(segment.id)}
                                                    style={{
                                                        background: 'none',
                                                        border: 'none',
                                                        color: '#f7768e',
                                                        cursor: 'pointer',
                                                        fontSize: '1.2rem',
                                                        padding: '0 4px'
                                                    }}
                                                >
                                                    x
                                                </button>
                                            </div>

                                            {/* Position Input */}
                                            <div style={{ marginBottom: '10px' }}>
                                                <label style={{ fontSize: '0.75rem', color: '#565f89', display: 'block', marginBottom: '4px' }}>
                                                    Start Position (1 - {seqLen - 2})
                                                </label>
                                                <input
                                                    type="number"
                                                    min={1}
                                                    max={seqLen - 2}
                                                    value={segment.position}
                                                    onChange={(e) => updateSegmentPosition(segment.id, e.target.value)}
                                                    style={{
                                                        width: '100%',
                                                        padding: '8px',
                                                        backgroundColor: '#2e3240',
                                                        border: '1px solid #3b3d4d',
                                                        borderRadius: '4px',
                                                        color: '#a9b1d6',
                                                        fontSize: '0.9rem',
                                                        boxSizing: 'border-box'
                                                    }}
                                                />
                                            </div>

                                            {/* Text Input */}
                                            <div style={{ marginBottom: '10px' }}>
                                                <label style={{ fontSize: '0.75rem', color: '#565f89', display: 'block', marginBottom: '4px' }}>
                                                    Text
                                                </label>
                                                <textarea
                                                    value={segment.text}
                                                    onChange={(e) => handleTextChange(segment.id, e.target.value)}
                                                    placeholder="Enter text..."
                                                    style={{
                                                        width: '100%',
                                                        minHeight: '60px',
                                                        padding: '8px',
                                                        backgroundColor: '#2e3240',
                                                        border: '1px solid #3b3d4d',
                                                        borderRadius: '4px',
                                                        color: '#a9b1d6',
                                                        fontSize: '0.9rem',
                                                        resize: 'vertical',
                                                        fontFamily: 'inherit',
                                                        boxSizing: 'border-box'
                                                    }}
                                                />
                                            </div>

                                            {/* Token Preview */}
                                            {segment.tokens.length > 0 && (
                                                <div>
                                                    <label style={{ fontSize: '0.75rem', color: '#565f89', display: 'block', marginBottom: '4px' }}>
                                                        Tokens ({segment.tokens.length}) - Positions {segment.position} to {segment.position + segment.tokens.length - 1}
                                                    </label>
                                                    <div style={{
                                                        display: 'flex',
                                                        flexWrap: 'wrap',
                                                        gap: '3px',
                                                        maxHeight: '80px',
                                                        overflowY: 'auto',
                                                        padding: '6px',
                                                        backgroundColor: '#2e3240',
                                                        borderRadius: '4px'
                                                    }}>
                                                        {segment.tokens.map((tok, tIdx) => (
                                                            <span
                                                                key={tIdx}
                                                                title={`ID: ${tok.id}, Pos: ${segment.position + tIdx}`}
                                                                style={{
                                                                    padding: '2px 5px',
                                                                    backgroundColor: '#3b3d4d',
                                                                    borderRadius: '3px',
                                                                    fontSize: '0.75rem',
                                                                    color: '#7dcfff',
                                                                    fontFamily: 'monospace',
                                                                    whiteSpace: 'pre'
                                                                }}
                                                            >
                                                                {normalizeTokenString(tok.text) || '\u00A0'}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {tokenizingId === segment.id && (
                                                <div style={{ fontSize: '0.75rem', color: '#565f89', marginTop: '5px' }}>
                                                    Tokenizing...
                                                </div>
                                            )}

                                            {isOverlapping && (
                                                <div style={{ fontSize: '0.75rem', color: '#f7768e', marginTop: '8px' }}>
                                                    Overlaps with another segment
                                                </div>
                                            )}
                                        </div>
                                    );
                                    })}

                                    {/* Add Segment Button */}
                                    <button
                                        onClick={addPromptSegment}
                                        style={{
                                            width: '100%',
                                            padding: '12px',
                                            backgroundColor: '#2e3240',
                                            border: '1px dashed #3b3d4d',
                                            borderRadius: '8px',
                                            color: '#7aa2f7',
                                            cursor: 'pointer',
                                            fontSize: '0.9rem',
                                            transition: 'background-color 0.2s'
                                        }}
                                        onMouseOver={(e) => e.target.style.backgroundColor = '#3b3d4d'}
                                        onMouseOut={(e) => e.target.style.backgroundColor = '#2e3240'}
                                    >
                                        + Add Text Segment
                                    </button>
                                </div>

                                {/* Sequence Preview (mini) */}
                                {promptSegments.length > 0 && (
                                    <div style={{ marginTop: '20px' }}>
                                        <label style={{ fontSize: '0.8rem', color: '#565f89', display: 'block', marginBottom: '8px' }}>
                                            Sequence Preview (showing filled regions)
                                        </label>
                                        <div style={{
                                            height: '30px',
                                            backgroundColor: '#2e3240',
                                            borderRadius: '4px',
                                            position: 'relative',
                                            overflow: 'hidden'
                                        }}>
                                            {/* Start token */}
                                            <div style={{
                                                position: 'absolute',
                                                left: '0%',
                                                width: `${(1 / seqLen) * 100}%`,
                                                height: '100%',
                                                backgroundColor: '#f7768e',
                                                minWidth: '2px'
                                            }} title="Start: <|endoftext|>" />

                                            {/* Segment regions */}
                                            {promptSegments.map((segment, idx) => {
                                                const startPct = (segment.position / seqLen) * 100;
                                                const widthPct = ((segment.tokenIds?.length || 0) / seqLen) * 100;
                                                if (widthPct === 0) return null;
                                                return (
                                                    <div
                                                        key={segment.id}
                                                        style={{
                                                            position: 'absolute',
                                                            left: `${startPct}%`,
                                                            width: `${Math.max(widthPct, 0.5)}%`,
                                                            height: '100%',
                                                            backgroundColor: '#7aa2f7',
                                                            minWidth: '2px'
                                                        }}
                                                        title={`Segment ${idx + 1}: positions ${segment.position}-${segment.position + (segment.tokenIds?.length || 1) - 1}`}
                                                    />
                                                );
                                            })}

                                            {/* End token */}
                                            <div style={{
                                                position: 'absolute',
                                                right: '0%',
                                                width: `${(1 / seqLen) * 100}%`,
                                                height: '100%',
                                                backgroundColor: '#f7768e',
                                                minWidth: '2px'
                                            }} title="End: <|endoftext|>" />
                                        </div>
                                        <div style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            fontSize: '0.7rem',
                                            color: '#565f89',
                                            marginTop: '4px'
                                        }}>
                                            <span>0</span>
                                            <span>{seqLen - 1}</span>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Progress Footer */}
            {history && (
                <div style={{ height: '4px', backgroundColor: '#1a1b26', width: '100%' }}>
                    <div style={{
                        height: '100%',
                        width: `${(currentStep / (history.length - 1)) * 100}%`,
                        backgroundColor: '#7aa2f7',
                        transition: 'width 0.1s linear'
                    }} />
                </div>
            )}
        </div>
    );
}
