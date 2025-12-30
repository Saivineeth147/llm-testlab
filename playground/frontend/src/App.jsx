import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './markdown.css'
import {
    Beaker, Settings, Key, Zap, Code, FileText, Play, Plus, X, Menu,
    ChevronDown, AlertCircle, CheckCircle, Search, Shield, Brain, Database, Sparkles, Languages, Terminal, Target,
    Loader2, Download, RefreshCw
} from 'lucide-react'

// ═══════════════════════════════════════════════════════════════════════════
// API Client
// ═══════════════════════════════════════════════════════════════════════════

const api = {
    async getProviders() {
        const res = await fetch('/api/providers')
        return res.json()
    },

    async fetchModels(provider, apiKey) {
        const res = await fetch('/api/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, api_key: apiKey })
        })
        if (!res.ok) {
            throw new Error('Failed to fetch models')
        }
        return res.json()
    },

    async testConnection(provider, apiKey, model) {
        const res = await fetch('/api/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, api_key: apiKey, model })
        })
        return res.json()
    },

    async evaluateText(data) {
        const res = await fetch('/api/evaluate/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
        if (!res.ok) {
            const error = await res.json()
            throw new Error(error.detail || 'Evaluation failed')
        }
        return res.json()
    },

    async evaluateCode(data) {
        const res = await fetch('/api/evaluate/code', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
        if (!res.ok) {
            const error = await res.json()
            throw new Error(error.detail || 'Code evaluation failed')
        }
        return res.json()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Metric Card Component
// ═══════════════════════════════════════════════════════════════════════════

function MetricCard({ label, value, description, isGood, icon: Icon, range }) {
    const numValue = typeof value === 'number' ? value : parseFloat(value) || 0
    const displayValue = typeof value === 'boolean'
        ? (value ? '✓' : '✗')
        : numValue.toFixed(2)

    const getStatus = () => {
        if (typeof value === 'boolean') return value ? 'metric-good' : 'metric-bad'
        if (isGood === 'lower') return numValue < 0.3 ? 'metric-good' : numValue < 0.6 ? 'metric-warning' : 'metric-bad'
        return numValue > 0.7 ? 'metric-good' : numValue > 0.4 ? 'metric-warning' : 'metric-bad'
    }

    const getIndicator = () => {
        if (typeof value === 'boolean') return null
        if (isGood === 'lower') return { text: '↓ Lower is better', color: 'var(--success)' }
        return { text: '↑ Higher is better', color: 'var(--success)' }
    }

    const indicator = getIndicator()

    return (
        <div className={`metric-card ${getStatus()}`}>
            {Icon && <div className="section-icon" style={{ marginBottom: '8px' }}><Icon size={24} /></div>}
            <div className="metric-label">{label}</div>
            <div className="metric-value">{displayValue}</div>
            <div className="progress-bar">
                <div
                    className="progress-fill"
                    style={{ width: `${Math.min(numValue * 100, 100)}%` }}
                />
            </div>
            <div className="metric-description">{description}</div>
            {indicator && (
                <div className="metric-indicator" style={{
                    fontSize: '0.7rem',
                    color: 'var(--text-muted)',
                    marginTop: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                }}>
                    <span style={{ color: indicator.color }}>{indicator.text}</span>
                    {range && <span>• Range: {range}</span>}
                </div>
            )}
        </div>
    )
}


// ═══════════════════════════════════════════════════════════════════════════
// Text List Input Component (supports long text entries)
// ═══════════════════════════════════════════════════════════════════════════

function TextListInput({ items, setItems, placeholder, label }) {
    const [input, setInput] = useState('')

    const addItem = () => {
        if (input.trim()) {
            setItems([...items, input.trim()])
            setInput('')
        }
    }

    const handleKeyDown = (e) => {
        // Ctrl+Enter or Cmd+Enter to add
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && input.trim()) {
            e.preventDefault()
            addItem()
        }
    }

    const removeItem = (index) => {
        setItems(items.filter((_, i) => i !== index))
    }

    return (
        <div className="text-list-container">
            {/* Existing items */}
            {items.map((item, index) => (
                <div key={index} className="text-list-item" style={{
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-md)',
                    padding: '8px 12px',
                    marginBottom: 8,
                    display: 'flex',
                    gap: 8,
                    alignItems: 'flex-start'
                }}>
                    <span style={{ flex: 1, fontSize: '0.85rem', whiteSpace: 'pre-wrap' }}>{item}</span>
                    <button
                        onClick={() => removeItem(index)}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            padding: 4,
                            flexShrink: 0
                        }}
                    >
                        <X size={14} />
                    </button>
                </div>
            ))}

            {/* Input area */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                <textarea
                    className="form-textarea"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    style={{
                        minHeight: 60,
                        flex: 1,
                        resize: 'vertical'
                    }}
                />
                <button
                    className="btn btn-secondary"
                    onClick={addItem}
                    disabled={!input.trim()}
                    style={{
                        flexShrink: 0,
                        padding: '8px 12px',
                        fontSize: '0.8rem',
                        height: 'fit-content'
                    }}
                >
                    <Plus size={14} />
                    Add
                </button>
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4, display: 'block' }}>
                Press Ctrl+Enter to add quickly
            </span>
        </div>
    )
}

// Simple TagInput for Knowledge Base (short entries)
function TagInput({ tags, setTags, placeholder }) {
    const [input, setInput] = useState('')

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && input.trim()) {
            e.preventDefault()
            setTags([...tags, input.trim()])
            setInput('')
        }
    }

    const removeTag = (index) => {
        setTags(tags.filter((_, i) => i !== index))
    }

    return (
        <div className="tag-input-container">
            {tags.map((tag, index) => (
                <span key={index} className="tag">
                    {tag.length > 50 ? tag.substring(0, 50) + '...' : tag}
                    <span className="tag-remove" onClick={() => removeTag(index)}>
                        <X size={14} />
                    </span>
                </span>
            ))}
            <input
                type="text"
                className="tag-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={tags.length === 0 ? placeholder : ''}
            />
        </div>
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Main App Component
// ═══════════════════════════════════════════════════════════════════════════

export default function App() {
    // Provider state
    const [providers, setProviders] = useState({})
    const [selectedProvider, setSelectedProvider] = useState('groq')
    const [apiKey, setApiKey] = useState('')
    const [selectedModel, setSelectedModel] = useState('')
    const [availableModels, setAvailableModels] = useState([])
    const [modelsLoading, setModelsLoading] = useState(false)
    const [temperature, setTemperature] = useState(0.7)
    const [topP, setTopP] = useState(0.9)
    const [maxTokens, setMaxTokens] = useState(1024)
    const [connectionStatus, setConnectionStatus] = useState(null)

    // Evaluation mode
    const [mode, setMode] = useState('text') // 'text', 'code', or 'docs'

    // Text evaluation state
    const [prompt, setPrompt] = useState('')
    const [paraphrases, setParaphrases] = useState([])
    const [adversarialPrompts, setAdversarialPrompts] = useState([])
    const [knowledgeBase, setKnowledgeBase] = useState([])
    const [runs, setRuns] = useState(3)

    // Code evaluation state
    const [codePrompt, setCodePrompt] = useState('')
    const [codeResponse, setCodeResponse] = useState('')
    const [referenceCode, setReferenceCode] = useState('')
    const [language, setLanguage] = useState('python')

    // Results
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [apiKeyError, setApiKeyError] = useState(null)
    const [textResults, setTextResults] = useState(null)
    const [codeResults, setCodeResults] = useState(null)

    // Mobile sidebar
    const [sidebarOpen, setSidebarOpen] = useState(false)

    // Load providers on mount
    useEffect(() => {
        api.getProviders().then(setProviders).catch(console.error)

        // Load saved API keys from localStorage
        const savedKeys = localStorage.getItem('llm-playground-keys')
        if (savedKeys) {
            const keys = JSON.parse(savedKeys)
            if (keys[selectedProvider]) {
                setApiKey(keys[selectedProvider])
            }
        }
    }, [])

    // Update model when provider changes
    useEffect(() => {
        if (providers[selectedProvider]) {
            setSelectedModel(providers[selectedProvider].default_model)
            setAvailableModels([]) // Clear models until API key is provided

            // Load saved key for this provider
            const savedKeys = localStorage.getItem('llm-playground-keys')
            if (savedKeys) {
                const keys = JSON.parse(savedKeys)
                setApiKey(keys[selectedProvider] || '')
            }
        }
    }, [selectedProvider, providers])

    // Fetch models when API key changes
    useEffect(() => {
        if (apiKey && apiKey.length > 10) {
            setModelsLoading(true)
            setApiKeyError(null)
            api.fetchModels(selectedProvider, apiKey)
                .then(data => {
                    setAvailableModels(data.models || [])
                    setConnectionStatus(null) // Reset connection status when key changes
                    // Set first model as default if current selection not in list
                    if (data.models?.length > 0) {
                        const modelIds = data.models.map(m => m.id)
                        if (!modelIds.includes(selectedModel)) {
                            setSelectedModel(data.models[0].id)
                        }
                        setApiKeyError(null)
                    } else {
                        setApiKeyError('No models available. Check your API key permissions.')
                    }
                })
                .catch(err => {
                    console.error('Failed to fetch models:', err)
                    setAvailableModels([])
                    setApiKeyError('Invalid API key or failed to fetch models')
                })
                .finally(() => setModelsLoading(false))
        } else if (apiKey && apiKey.length > 0 && apiKey.length <= 10) {
            setApiKeyError('API key seems too short')
        } else {
            setApiKeyError(null)
            setAvailableModels([])
        }
    }, [apiKey, selectedProvider])

    // Save API key to localStorage
    const saveApiKey = () => {
        const savedKeys = JSON.parse(localStorage.getItem('llm-playground-keys') || '{}')
        savedKeys[selectedProvider] = apiKey
        localStorage.setItem('llm-playground-keys', JSON.stringify(savedKeys))
    }

    // Test connection
    const testConnection = async () => {
        setConnectionStatus('testing')
        setApiKeyError(null)
        try {
            const result = await api.testConnection(selectedProvider, apiKey, selectedModel)
            if (result.success) {
                setConnectionStatus('success')
                saveApiKey()
            } else {
                setConnectionStatus('error')
                setApiKeyError(result.detail || 'Connection failed. Check your API key.')
            }
        } catch (err) {
            setConnectionStatus('error')
            setApiKeyError('Connection failed: ' + (err.message || 'Unknown error'))
        }
    }

    // Run text evaluation
    const runTextEvaluation = async () => {
        if (!apiKey) {
            setError('Please enter an API key first')
            return
        }
        if (!prompt.trim()) {
            setError('Please enter a prompt')
            return
        }

        setLoading(true)
        setError(null)

        try {
            const result = await api.evaluateText({
                provider_config: {
                    provider: selectedProvider,
                    api_key: apiKey,
                    model: selectedModel,
                    temperature,
                    max_tokens: maxTokens
                },
                prompt: prompt.trim(),
                paraphrases: paraphrases.length > 0 ? paraphrases : null,
                adversarial_prompts: adversarialPrompts.length > 0 ? adversarialPrompts : null,
                knowledge_base: knowledgeBase.length > 0 ? knowledgeBase : null,
                runs
            })
            setTextResults(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // Run code evaluation
    const runCodeEvaluation = async () => {
        if (!codePrompt.trim() && !codeResponse.trim()) {
            setError('Please enter a prompt or code to evaluate')
            return
        }

        setLoading(true)
        setError(null)

        try {
            const result = await api.evaluateCode({
                provider_config: {
                    provider: selectedProvider,
                    api_key: apiKey,
                    model: selectedModel,
                    temperature,
                    max_tokens: maxTokens
                },
                prompt: codePrompt.trim(),
                code_response: codeResponse.trim() || null,
                reference_code: referenceCode.trim() || null,
                language
            })
            setCodeResults(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // Download results as JSON
    const downloadResults = () => {
        const results = mode === 'text' ? textResults : codeResults
        if (!results) return

        const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `llm - evaluation - ${mode} -${Date.now()}.json`
        a.click()
    }

    return (
        <div className="app-container">
            {/* Mobile Header */}
            <div className="mobile-header">
                <button className="menu-toggle" onClick={() => setSidebarOpen(true)}>
                    <Menu size={24} />
                </button>
                <div className="logo" style={{ flex: 1 }}>
                    <div className="logo-icon" style={{ width: 32, height: 32 }}>
                        <Beaker size={18} />
                    </div>
                    <span className="logo-text" style={{ fontSize: '1rem' }}>LLM Playground</span>
                </div>
            </div>

            {/* Sidebar Overlay */}
            <div
                className={`sidebar - overlay ${sidebarOpen ? 'visible' : ''} `}
                onClick={() => setSidebarOpen(false)}
            />

            {/* ═══════════════════════════════════════════════════════════════════════
          Sidebar
          ═══════════════════════════════════════════════════════════════════════ */}
            <aside className={`sidebar ${sidebarOpen ? 'open' : ''} `}>
                {/* Close button for mobile */}
                <button
                    className="menu-toggle"
                    onClick={() => setSidebarOpen(false)}
                    style={{
                        position: 'absolute',
                        top: 16,
                        right: 16,
                        display: 'none'
                    }}
                >
                    <X size={20} />
                </button>

                {/* Logo */}
                <div className="logo">
                    <div className="logo-icon">
                        <Beaker size={22} />
                    </div>
                    <span className="logo-text">LLM Playground</span>
                </div>

                <div className="divider" />

                {/* Provider Selection */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">
                            <Zap size={14} style={{ marginRight: 6 }} />
                            Provider
                        </span>
                    </div>

                    <div className="form-group">
                        <select
                            className="form-select"
                            value={selectedProvider}
                            onChange={(e) => setSelectedProvider(e.target.value)}
                        >
                            {Object.entries(providers).map(([key, provider]) => (
                                <option key={key} value={key}>{provider.name}</option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group" style={{ marginTop: 12 }}>
                        <label className="form-label">
                            Model {modelsLoading && <Loader2 size={12} className="loading-spinner" style={{ marginLeft: 4 }} />}
                            {availableModels.length > 0 && (
                                <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
                                    {' '}({availableModels.length} available)
                                </span>
                            )}
                        </label>
                        <select
                            className="form-select"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            disabled={modelsLoading}
                        >
                            {availableModels.length > 0 ? (
                                availableModels.map(model => (
                                    <option key={model.id} value={model.id}>
                                        {model.name || model.id}
                                    </option>
                                ))
                            ) : (
                                <option value={providers[selectedProvider]?.default_model}>
                                    {providers[selectedProvider]?.default_model || 'Enter API key to load models'}
                                </option>
                            )}
                        </select>
                    </div>
                </div>

                {/* API Key */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">
                            <Key size={14} style={{ marginRight: 6 }} />
                            API Key
                        </span>
                        {connectionStatus === 'success' && (
                            <span className="badge badge-success">
                                <CheckCircle size={12} /> Connected
                            </span>
                        )}
                    </div>

                    <div className="form-group">
                        <input
                            type="password"
                            className="form-input"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            placeholder="Paste your API key here"
                            style={apiKeyError ? { borderColor: 'var(--error)' } : {}}
                        />
                        {apiKeyError ? (
                            <span style={{
                                fontSize: '0.7rem',
                                color: 'var(--error)',
                                marginTop: 4,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 4
                            }}>
                                <AlertCircle size={12} />
                                {apiKeyError}
                            </span>
                        ) : (
                            <span style={{
                                fontSize: '0.7rem',
                                color: availableModels.length > 0 ? 'var(--success)' : 'var(--text-muted)',
                                marginTop: 4,
                                display: 'block'
                            }}>
                                {availableModels.length > 0
                                    ? `✓ ${availableModels.length} models loaded`
                                    : 'Models will auto-load when you enter a valid key'}
                            </span>
                        )}
                    </div>

                    <button
                        className="btn btn-secondary"
                        style={{ marginTop: 8, width: '100%' }}
                        onClick={testConnection}
                        disabled={!apiKey || modelsLoading}
                    >
                        {connectionStatus === 'testing' ? (
                            <Loader2 size={16} className="loading-spinner" />
                        ) : connectionStatus === 'error' ? (
                            <>
                                <AlertCircle size={14} />
                                Retry Connection
                            </>
                        ) : connectionStatus === 'success' ? (
                            <>
                                <CheckCircle size={14} />
                                Connected!
                            </>
                        ) : (
                            <>
                                <Zap size={14} />
                                Test Connection
                            </>
                        )}
                    </button>

                    {providers[selectedProvider]?.api_key_url && (
                        <a
                            href={providers[selectedProvider].api_key_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{
                                fontSize: '0.75rem',
                                color: 'var(--accent-secondary)',
                                marginTop: 8,
                                display: 'block'
                            }}
                        >
                            🔑 Get your {providers[selectedProvider]?.name} API key →
                        </a>
                    )}
                </div>

                {/* Settings */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">
                            <Settings size={14} style={{ marginRight: 6 }} />
                            Settings
                        </span>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Temperature: {temperature}</label>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            style={{ width: '100%' }}
                        />
                    </div>

                    <div className="form-group" style={{ marginTop: 12 }}>
                        <label className="form-label">Top-P: {topP}</label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={topP}
                            onChange={(e) => setTopP(parseFloat(e.target.value))}
                            style={{ width: '100%' }}
                        />
                    </div>

                    <div className="form-group" style={{ marginTop: 12 }}>
                        <label className="form-label">Max Tokens</label>
                        <input
                            type="number"
                            className="form-input"
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                            min={1}
                            max={8192}
                        />
                    </div>
                </div>
            </aside>

            {/* ═══════════════════════════════════════════════════════════════════════
          Main Content
          ═══════════════════════════════════════════════════════════════════════ */}
            <main className="main-content">
                {/* Header */}
                <div className="header">
                    <h1 className="header-title">
                        <Sparkles size={24} style={{ marginRight: 8, color: 'var(--accent-primary)' }} />
                        Model Evaluation
                    </h1>
                    <div className="header-actions">
                        {(textResults || codeResults) && (
                            <button className="btn btn-secondary" onClick={downloadResults}>
                                <Download size={16} />
                                Export JSON
                            </button>
                        )}
                    </div>
                </div>

                {/* Mode Tabs */}
                <div className="tabs">
                    <button
                        className={`tab ${mode === 'text' ? 'active' : ''} `}
                        onClick={() => setMode('text')}
                    >
                        <FileText size={18} />
                        <span className="tab-label">Text</span>
                    </button>
                    <button
                        className={`tab ${mode === 'code' ? 'active' : ''} `}
                        onClick={() => setMode('code')}
                    >
                        <Code size={18} />
                        <span className="tab-label">Code</span>
                    </button>
                    <a
                        className="tab"
                        href="https://github.com/Saivineeth147/llm-testlab#readme"
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ textDecoration: 'none' }}
                    >
                        <FileText size={18} />
                        <span className="tab-label">Docs ↗</span>
                    </a>
                </div>

                {/* Error Alert */}
                {error && (
                    <div className="alert alert-error">
                        <AlertCircle size={20} />
                        <div>{error}</div>
                    </div>
                )}

                {/* ═══════════════════════════════════════════════════════════════════
            Text Evaluation Mode
            ═══════════════════════════════════════════════════════════════════ */}
                {mode === 'text' && (
                    <>
                        {/* Prompt Input */}
                        <section className="section">
                            <div className="section-header">
                                <Brain className="section-icon" size={20} />
                                <h2 className="section-title">Prompt</h2>
                            </div>
                            <textarea
                                className="form-textarea"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="Enter your prompt to evaluate..."
                                style={{ minHeight: 100 }}
                            />
                        </section>

                        {/* Paraphrases */}
                        <section className="section">
                            <div className="section-header">
                                <RefreshCw className="section-icon" size={20} />
                                <h2 className="section-title">Paraphrases (for SRI)</h2>
                            </div>
                            <TextListInput
                                items={paraphrases}
                                setItems={setParaphrases}
                                placeholder="Add paraphrased versions of your prompt. These should convey the same meaning in different words..."
                            />
                        </section>

                        {/* Adversarial Prompts */}
                        <section className="section">
                            <div className="section-header">
                                <Shield className="section-icon" size={20} />
                                <h2 className="section-title">Adversarial Prompts (for SVE)</h2>
                            </div>
                            <TextListInput
                                items={adversarialPrompts}
                                setItems={setAdversarialPrompts}
                                placeholder="Add adversarial prompts to test safety. These can be jailbreaks, harmful requests, or attempts to bypass restrictions..."
                            />
                        </section>

                        {/* Knowledge Base */}
                        <section className="section">
                            <div className="section-header">
                                <Database className="section-icon" size={20} />
                                <h2 className="section-title">Knowledge Base (for HSI & KBC)</h2>
                            </div>
                            <TextListInput
                                items={knowledgeBase}
                                setItems={setKnowledgeBase}
                                placeholder="Add factual statements the model should know. Used to detect hallucinations and measure knowledge coverage..."
                            />
                        </section>

                        {/* Run Button */}
                        <button
                            className="btn btn-primary"
                            onClick={runTextEvaluation}
                            disabled={loading || !apiKey}
                            style={{ marginBottom: 24 }}
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={18} className="loading-spinner" />
                                    Evaluating...
                                </>
                            ) : (
                                <>
                                    <Play size={18} />
                                    Run Text Evaluation
                                </>
                            )}
                        </button>

                        {/* Text Results */}
                        {textResults && (
                            <section className="section">
                                <div className="section-header">
                                    <Target className="section-icon" size={20} />
                                    <h2 className="section-title">Results</h2>
                                    <span className="badge badge-info" style={{ marginLeft: 'auto' }}>
                                        {textResults.model}
                                    </span>
                                </div>

                                <div className="metrics-grid">
                                    {textResults.metrics.HSI && (
                                        <MetricCard
                                            label="HSI"
                                            value={textResults.metrics.HSI.value}
                                            description="Measures factual deviation from knowledge base"
                                            isGood="lower"
                                            icon={Brain}
                                            range="0-1"
                                        />
                                    )}
                                    {textResults.metrics.CSS && (
                                        <MetricCard
                                            label="CSS"
                                            value={textResults.metrics.CSS.value}
                                            description="Output stability across multiple runs"
                                            isGood="higher"
                                            icon={RefreshCw}
                                            range="0-1"
                                        />
                                    )}
                                    {textResults.metrics.SRI && (
                                        <MetricCard
                                            label="SRI"
                                            value={textResults.metrics.SRI.value}
                                            description="Invariance to input paraphrasing"
                                            isGood="higher"
                                            icon={Target}
                                            range="0-1"
                                        />
                                    )}
                                    {textResults.metrics.SVE && (
                                        <MetricCard
                                            label="SVE"
                                            value={textResults.metrics.SVE.value}
                                            description="Rate of unsafe/harmful responses"
                                            isGood="lower"
                                            icon={Shield}
                                            range="0-1"
                                        />
                                    )}
                                    {textResults.metrics.KBC && (
                                        <MetricCard
                                            label="KBC"
                                            value={textResults.metrics.KBC.value}
                                            description="Alignment with knowledge base facts"
                                            isGood="higher"
                                            icon={Database}
                                            range="0-1"
                                        />
                                    )}
                                </div>

                                {/* Response Preview */}
                                {textResults.metrics.HSI?.answer && (
                                    <div className="results-panel" style={{ marginTop: 24 }}>
                                        <div className="result-item">
                                            <div className="result-label">Model Response</div>
                                            <div className="result-value markdown-content">
                                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                    {textResults.metrics.HSI.answer}
                                                </ReactMarkdown>
                                            </div>
                                        </div>
                                        {textResults.metrics.HSI.closest_fact && (
                                            <div className="result-item">
                                                <div className="result-label">Closest Fact in KB</div>
                                                <div className="result-value">{textResults.metrics.HSI.closest_fact}</div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </section>
                        )}
                    </>
                )}

                {/* ═══════════════════════════════════════════════════════════════════
            Code Evaluation Mode
            ═══════════════════════════════════════════════════════════════════ */}
                {mode === 'code' && (
                    <>
                        {/* Code Prompt */}
                        <section className="section">
                            <div className="section-header">
                                <Brain className="section-icon" size={20} />
                                <h2 className="section-title">Code Generation Prompt</h2>
                            </div>
                            <textarea
                                className="form-textarea"
                                value={codePrompt}
                                onChange={(e) => setCodePrompt(e.target.value)}
                                placeholder="Describe what code you want to generate..."
                                style={{ minHeight: 80 }}
                            />
                        </section>

                        {/* Language Selection */}
                        <section className="section">
                            <div className="form-group" style={{ maxWidth: 200 }}>
                                <label className="form-label">Language</label>
                                <select
                                    className="form-select"
                                    value={language}
                                    onChange={(e) => setLanguage(e.target.value)}
                                >
                                    <option value="python">Python</option>
                                    <option value="javascript">JavaScript</option>
                                    <option value="typescript">TypeScript</option>
                                    <option value="java">Java</option>
                                    <option value="cpp">C++</option>
                                    <option value="go">Go</option>
                                    <option value="rust">Rust</option>
                                    <option value="ruby">Ruby</option>
                                    <option value="php">PHP</option>
                                </select>
                            </div>
                        </section>

                        {/* Code Input */}
                        <section className="section">
                            <div className="section-header">
                                <Code className="section-icon" size={20} />
                                <h2 className="section-title">Code to Evaluate (optional)</h2>
                            </div>
                            <div className="code-editor">
                                <div className="code-header">
                                    <div className="code-dots">
                                        <div className="code-dot red"></div>
                                        <div className="code-dot yellow"></div>
                                        <div className="code-dot green"></div>
                                    </div>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        {language}
                                    </span>
                                </div>
                                <div className="code-content">
                                    <textarea
                                        className="code-textarea"
                                        value={codeResponse}
                                        onChange={(e) => setCodeResponse(e.target.value)}
                                        placeholder="Paste code here, or leave empty to generate from prompt..."
                                    />
                                </div>
                            </div>
                        </section>

                        {/* Reference Code */}
                        <section className="section">
                            <div className="section-header">
                                <FileText className="section-icon" size={20} />
                                <h2 className="section-title">Reference Code (optional)</h2>
                            </div>
                            <div className="code-editor">
                                <div className="code-header">
                                    <div className="code-dots">
                                        <div className="code-dot red"></div>
                                        <div className="code-dot yellow"></div>
                                        <div className="code-dot green"></div>
                                    </div>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        reference
                                    </span>
                                </div>
                                <div className="code-content">
                                    <textarea
                                        className="code-textarea"
                                        value={referenceCode}
                                        onChange={(e) => setReferenceCode(e.target.value)}
                                        placeholder="Paste reference solution for semantic comparison..."
                                        style={{ minHeight: 100 }}
                                    />
                                </div>
                            </div>
                        </section>

                        {/* Run Button */}
                        <button
                            className="btn btn-primary"
                            onClick={runCodeEvaluation}
                            disabled={loading || !apiKey}
                            style={{ marginBottom: 24 }}
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={18} className="loading-spinner" />
                                    Evaluating...
                                </>
                            ) : (
                                <>
                                    <Play size={18} />
                                    Run Code Evaluation
                                </>
                            )}
                        </button>

                        {/* Code Results */}
                        {codeResults && (
                            <section className="section">
                                <div className="section-header">
                                    <Target className="section-icon" size={20} />
                                    <h2 className="section-title">Code Evaluation Results</h2>
                                    <span className="badge badge-info" style={{ marginLeft: 'auto' }}>
                                        {codeResults.model}
                                    </span>
                                </div>

                                <div className="metrics-grid">
                                    <MetricCard
                                        label="Overall Score"
                                        value={codeResults.metrics.overall_score / 100}
                                        description="Combined score of all code quality metrics"
                                        isGood="higher"
                                        icon={Sparkles}
                                        range="0-100"
                                    />
                                    <MetricCard
                                        label="Syntax"
                                        value={codeResults.metrics.syntax_valid}
                                        description={codeResults.metrics.syntax_valid ? 'Valid' : 'Invalid'}
                                        icon={CheckCircle}
                                    />
                                    <MetricCard
                                        label="Quality"
                                        value={codeResults.metrics.quality_score / 100}
                                        description="Code structure, readability & best practices"
                                        isGood="higher"
                                        icon={Target}
                                        range="0-100"
                                    />
                                    <MetricCard
                                        label="Security"
                                        value={codeResults.metrics.is_secure}
                                        description={codeResults.metrics.is_secure ? 'Secure' : 'Issues Found'}
                                        icon={Shield}
                                    />
                                    {
                                        codeResults.metrics.semantic_similarity !== null && (
                                            <MetricCard
                                                label="Semantic Match"
                                                value={codeResults.metrics.semantic_similarity}
                                                description="Code similarity to reference solution"
                                                isGood="higher"
                                                icon={Brain}
                                                range="0-1"
                                            />
                                        )
                                    }
                                </div >

                                {/* Quality Breakdown */}
                                {
                                    codeResults.metrics.quality_metrics && (
                                        <div className="results-panel" style={{ marginTop: 16 }}>
                                            <div className="result-label" style={{ marginBottom: 8 }}>Quality Breakdown</div>
                                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Lines of Code</span>
                                                    <div style={{ fontWeight: 600 }}>{codeResults.metrics.quality_metrics.lines_of_code || 0}</div>
                                                </div>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Functions</span>
                                                    <div style={{ fontWeight: 600 }}>{codeResults.metrics.quality_metrics.num_functions || 0}</div>
                                                </div>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Classes</span>
                                                    <div style={{ fontWeight: 600 }}>{codeResults.metrics.quality_metrics.num_classes || 0}</div>
                                                </div>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Complexity</span>
                                                    <div style={{ fontWeight: 600 }}>{codeResults.metrics.quality_metrics.cyclomatic_complexity || 0}</div>
                                                </div>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Has Comments</span>
                                                    <div style={{ fontWeight: 600, color: codeResults.metrics.quality_metrics.has_comments ? 'var(--success)' : 'var(--text-muted)' }}>
                                                        {codeResults.metrics.quality_metrics.has_comments ? '✓ Yes' : '✗ No'}
                                                    </div>
                                                </div>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Has Docstring</span>
                                                    <div style={{ fontWeight: 600, color: codeResults.metrics.quality_metrics.has_docstring ? 'var(--success)' : 'var(--text-muted)' }}>
                                                        {codeResults.metrics.quality_metrics.has_docstring ? '✓ Yes' : '✗ No'}
                                                    </div>
                                                </div>
                                                <div className="result-item" style={{ padding: 8 }}>
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Error Handling</span>
                                                    <div style={{ fontWeight: 600, color: codeResults.metrics.quality_metrics.has_error_handling ? 'var(--success)' : 'var(--text-muted)' }}>
                                                        {codeResults.metrics.quality_metrics.has_error_handling ? '✓ Yes' : '✗ No'}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }

                                {/* Syntax Error Display */}
                                {
                                    codeResults.metrics.syntax_error && (
                                        <div className="alert alert-error" style={{ marginTop: 16 }}>
                                            <AlertCircle size={20} />
                                            <div>
                                                <strong>Syntax Error:</strong>
                                                <pre style={{ marginTop: 4, fontSize: '0.8rem', whiteSpace: 'pre-wrap' }}>
                                                    {codeResults.metrics.syntax_error}
                                                </pre>
                                            </div>
                                        </div>
                                    )
                                }

                                {/* Generated Code */}
                                {
                                    codeResults.code && (
                                        <div className="results-panel" style={{ marginTop: 24 }}>
                                            <div className="result-label" style={{ marginBottom: 12 }}>
                                                Generated Code
                                                {codeResults.raw_response && (
                                                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginLeft: 8 }}>
                                                        (extracted from markdown)
                                                    </span>
                                                )}
                                            </div>
                                            <div className="code-editor">
                                                <div className="code-header">
                                                    <div className="code-dots">
                                                        <div className="code-dot red"></div>
                                                        <div className="code-dot yellow"></div>
                                                        <div className="code-dot green"></div>
                                                    </div>
                                                </div>
                                                <div className="code-content">
                                                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                                                        {codeResults.code}
                                                    </pre>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                }

                                {/* Vulnerabilities */}
                                {
                                    codeResults.metrics.vulnerabilities?.length > 0 && (
                                        <div className="alert alert-error" style={{ marginTop: 16 }}>
                                            <AlertCircle size={20} />
                                            <div>
                                                <strong>Security Issues Found ({codeResults.metrics.vulnerabilities.length}):</strong>
                                                <ul style={{ marginTop: 8, paddingLeft: 20 }}>
                                                    {codeResults.metrics.vulnerabilities.map((v, i) => (
                                                        <li key={i} style={{ marginBottom: 4 }}>
                                                            <strong style={{ color: v.severity === 'HIGH' ? 'var(--error)' : 'var(--warning)' }}>
                                                                [{v.severity}]
                                                            </strong> {v.type}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>
                                    )
                                }
                            </section >
                        )}
                    </>
                )}
            </main >
        </div >
    )
}

