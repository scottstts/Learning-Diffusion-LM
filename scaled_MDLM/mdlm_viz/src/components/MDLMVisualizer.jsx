import React, { useState, useEffect, useRef } from 'react';

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
    const [isLoading, setIsLoading] = useState(false);
    const [temperature, setTemperature] = useState(1.0);
    const [stepCountConfig] = useState(50);
    const [error, setError] = useState(null);

    const animationFrameRef = useRef();
    const lastFrameTimeRef = useRef(0);
    const FPS = 10; // Frames per second
    const frameInterval = 1000 / FPS;

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
                            setIsPlaying(false); // Stop at end
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

    const handleGenerate = async () => {
        setIsLoading(true);
        setError(null);
        setHistory(null);
        setCurrentStep(0);
        setIsPlaying(false);

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    temperature: temperature,
                    steps: stepCountConfig
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

            // Auto-play after loading
            setIsPlaying(true);
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
        // Basic cleanup similar to python
        return normalizeTokenString(raw);
    };

    // Helper to merge tokens into word groups for display
    const getDisplayGroups = (tokens) => {
        if (!tokens) return [];

        const groups = [];
        let currentGroup = [];

        tokens.forEach((id) => {
            const rawToken = vocab[id] || '';
            const isMask = id === maskTokenId;
            const isEndOfText = rawToken === '<|endoftext|>';
            const newlineCount = (rawToken.match(/\r?\n/g) || []).length;

            // Always isolate special tokens.
            if (isEndOfText) {
                if (currentGroup.length > 0) {
                    groups.push(currentGroup);
                    currentGroup = [];
                }

                groups.push([
                    {
                        id,
                        text: '',
                        isMask: false,
                        isEndOfText: true
                    }
                ]);
                return;
            }

            // Mask tokens should never merge with anything; render as standalone '?' cells.
            if (isMask) {
                if (currentGroup.length > 0) {
                    groups.push(currentGroup);
                    currentGroup = [];
                }

                groups.push([
                    {
                        id,
                        text: '?',
                        isMask: true,
                        isEndOfText: false,
                        isBreak: false
                    }
                ]);
                return;
            }

            // Newlines should not be displayed as '↵'; treat them as hard line breaks.
            if (newlineCount > 0) {
                if (currentGroup.length > 0) {
                    groups.push(currentGroup);
                    currentGroup = [];
                }

                for (let i = 0; i < newlineCount; i++) {
                    groups.push([
                        {
                            id,
                            text: '',
                            isMask: false,
                            isEndOfText: false,
                            isBreak: true
                        }
                    ]);
                }

                const remainderRaw = rawToken.replace(/\r?\n/g, '');
                const remainderText = normalizeTokenString(remainderRaw);
                if (remainderText.trim() !== '') {
                    currentGroup.push({
                        id,
                        text: remainderText,
                        isMask: false,
                        isEndOfText: false,
                        isBreak: false
                    });
                }

                return;
            }

            // Logic to start a new word group
            // 1. Explicit space prefix
            // 2. Starts with specific punctuation like "
            // 3. Is <|endoftext|> (always isolate)
            const isNewWord = rawToken.startsWith(' ') ||
                rawToken.startsWith('Ġ') ||
                rawToken.startsWith('"') ||
                rawToken.startsWith('“');

            if (isNewWord && currentGroup.length > 0) {
                groups.push(currentGroup);
                currentGroup = [];
            }

            const tokenDisplay = getTokenDisplay(id);

            // Skip whitespace-only tokens entirely (these typically represent standalone space markers
            // and render as ultra-thin "stripes" in the grid).
            if (tokenDisplay.trim() === '') {
                if (currentGroup.length > 0) {
                    groups.push(currentGroup);
                    currentGroup = [];
                }
                return;
            }

            currentGroup.push({
                id,
                text: tokenDisplay, // Already normalized via getTokenDisplay
                isMask,
                // Removed wasJustUnmasked logic as requested
                isEndOfText,
                isBreak: false
            });
        });
        if (currentGroup.length > 0) groups.push(currentGroup);

        return groups;
    };

    const currentTokens = history ? history[currentStep] : [];
    const groups = getDisplayGroups(currentTokens);

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            backgroundColor: '#1a1b26',
            color: '#a9b1d6',
            overflow: 'hidden'
        }}>
            {/* Header */}
            <div style={{
                padding: '20px 30px',
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
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                        <label style={{ fontSize: '0.8rem' }}>Temperature: {temperature}</label>
                        <input
                            type="range"
                            min="0.1"
                            max="2.0"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            disabled={isLoading || isPlaying}
                        />
                    </div>

                    <button
                        onClick={isPlaying ? () => setIsPlaying(false) : () => setIsPlaying(true)}
                        disabled={!history || isLoading}
                        style={{ minWidth: '80px' }}
                    >
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>

                    <button
                        onClick={handleGenerate}
                        disabled={isLoading && !history} // Allow regen
                        style={{
                            backgroundColor: '#7aa2f7',
                            color: '#1a1b26',
                            fontWeight: 'bold',
                            opacity: isLoading ? 0.7 : 1
                        }}
                    >
                        {isLoading ? 'Running...' : 'New Batch'}
                    </button>
                </div>
            </div>

            {/* Main Content Area */}
            <div style={{
                flex: 1,
                minHeight: 0,
                overflowY: 'auto',
                padding: '30px',
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
                        Press "New Batch" to start generation.
                    </div>
                )}

                {groups.map((group, gIdx) => {
                    const isBreakGroup = group.length === 1 && (group[0].isEndOfText || group[0].isBreak);
                    if (isBreakGroup) {
                        return (
                            <div
                                key={gIdx}
                                style={{ flexBasis: '100%', width: '100%', height: '20px' }}
                            />
                        );
                    }

                    const mergedTextRaw = group.map((t) => (t.isMask ? '?' : t.text)).join('');
                    const mergedText = mergedTextRaw.replace(/^\s+/, '');
                    if (mergedText.length === 0) return null;

                    // Style logic (word-level)
                    let bg = CELL_BG_COLOR;
                    let color = TEXT_COLOR;
                    let border = '1px solid transparent';

                    // Mask groups are always single-token groups by construction.
                    const isMaskGroup = group.length === 1 && group[0].isMask;
                    if (isMaskGroup) {
                        bg = '#2a2635'; // MASK BG
                        color = MASK_COLOR;
                        border = '1px solid #45354a';
                    }

                    return (
                        <div
                            key={gIdx}
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
