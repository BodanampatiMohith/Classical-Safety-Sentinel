import React, { useState, useRef, useEffect, useMemo } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Chart utility (using simple SVG)
const RiskChart = ({ data, height = 150 }) => {
  if (!data || data.length === 0) return null;
  
  const maxRisk = Math.max(...data, 1);
  const width = Math.min(300, data.length * 3);
  const padding = 20;
  const chartWidth = width - 2 * padding;
  const chartHeight = height - 2 * padding;
  
  return (
    <svg width={width} height={height} className="risk-chart">
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((val, i) => (
        <line
          key={`grid-${i}`}
          x1={padding}
          y1={padding + chartHeight * (1 - val)}
          x2={width - padding}
          y2={padding + chartHeight * (1 - val)}
          stroke="#eee"
          strokeWidth="1"
        />
      ))}
      
      {/* Critical threshold */}
      <line
        x1={padding}
        y1={padding + chartHeight * (1 - 0.7)}
        x2={width - padding}
        y2={padding + chartHeight * (1 - 0.7)}
        stroke="#ff3333"
        strokeWidth="2"
        strokeDasharray="5,5"
        opacity="0.5"
      />
      
      {/* Warning threshold */}
      <line
        x1={padding}
        y1={padding + chartHeight * (1 - 0.4)}
        x2={width - padding}
        y2={padding + chartHeight * (1 - 0.4)}
        stroke="#ffa500"
        strokeWidth="2"
        strokeDasharray="5,5"
        opacity="0.5"
      />
      
      {/* Data bars */}
      {data.map((value, idx) => {
        const x = padding + (idx / data.length) * chartWidth;
        const barHeight = chartHeight * (value / maxRisk);
        const color = value >= 0.7 ? '#ff3333' : value >= 0.4 ? '#ffa500' : '#00cc00';
        
        return (
          <rect
            key={`bar-${idx}`}
            x={x}
            y={padding + chartHeight - barHeight}
            width={Math.max(1, chartWidth / data.length - 1)}
            height={barHeight}
            fill={color}
            opacity="0.7"
          />
        );
      })}
      
      {/* Axes */}
      <line x1={padding} y1={padding} x2={padding} y2={padding + chartHeight} stroke="#333" strokeWidth="2" />
      <line x1={padding} y1={padding + chartHeight} x2={width - padding} y2={padding + chartHeight} stroke="#333" strokeWidth="2" />
    </svg>
  );
};

// Factor breakdown visualization
const FactorBreakdown = ({ factors }) => {
  if (!factors) return null;
  
  const total = Object.values(factors).reduce((a, b) => a + Math.abs(b), 0) || 1;
  
  return (
    <div className="factor-breakdown">
      <h4>Risk Factor Contribution</h4>
      <div className="factors-grid">
        {Object.entries(factors).map(([key, value]) => (
          <div key={key} className="factor-item">
            <div className="factor-label">{key.replace(/_/g, ' ')}</div>
            <div className="factor-bar">
              <div
                className="factor-fill"
                style={{
                  width: `${Math.abs(value) / total * 100}%`,
                  backgroundColor: value > 0 ? '#ff6b6b' : '#4CAF50'
                }}
              />
            </div>
            <div className="factor-value">{(value / total * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Traffic Signal Indicator Component
const TrafficSignal = ({ level, riskScore, explanation }) => {
  const getLightColor = (safetyLevel) => {
    switch (safetyLevel) {
      case 'CRITICAL':
        return '#ff3333';
      case 'WARNING':
        return '#ffa500';
      case 'SAFE':
        return '#00cc00';
      default:
        return '#cccccc';
    }
  };
  
  const text = level || 'MONITORING';
  const color = getLightColor(text);
  
  return (
    <div className="traffic-signal-widget">
      <div className="signal-lights">
        <div
          className={`signal-light red ${text === 'CRITICAL' ? 'active' : ''}`}
          style={{ backgroundColor: text === 'CRITICAL' ? '#ff3333' : '#ffcccc' }}
        />
        <div
          className={`signal-light yellow ${text === 'WARNING' ? 'active' : ''}`}
          style={{ backgroundColor: text === 'WARNING' ? '#ffa500' : '#ffe6cc' }}
        />
        <div
          className={`signal-light green ${text === 'SAFE' ? 'active' : ''}`}
          style={{ backgroundColor: text === 'SAFE' ? '#00cc00' : '#ccffcc' }}
        />
      </div>
      <div className="signal-status">
        <div className="status-label" style={{ color }}>
          {text}
        </div>
        <div className="status-score">Risk: {(riskScore * 100).toFixed(1)}%</div>
        <div className="status-explanation">{explanation || 'Click to analyze'}</div>
      </div>
    </div>
  );
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [liveRiskData, setLiveRiskData] = useState([]);
  const [currentSafetyLevel, setCurrentSafetyLevel] = useState('SAFE');
  const [currentRiskScore, setCurrentRiskScore] = useState(0);
  const videoCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Fetch system stats on load
  useEffect(() => {
    fetchStats();
    fetchEvents();
    const interval = setInterval(() => {
      fetchStats();
      fetchEvents();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchEvents = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/events`);
      setEvents(response.data.events || []);
    } catch (error) {
      console.error('Error fetching events:', error);
    }
  };

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a video file');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/infer_clip`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      });

      setResults(response.data);
      fetchStats();
      fetchEvents();
      setActiveTab('results');

      if (response.data.annotated_video_path) {
        setTimeout(() => {
          playAnnotatedVideo(response.data.video_id);
        }, 1000);
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error processing video: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const playAnnotatedVideo = async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/download/${videoId}`, {
        responseType: 'blob',
      });
      const url = URL.createObjectURL(response.data);
      const video = document.createElement('video');
      video.src = url;
      video.controls = true;
      video.width = 640;
      video.height = 480;
      
      const container = document.getElementById('videoContainer');
      if (container) {
        container.innerHTML = '';
        container.appendChild(video);
      }
    } catch (error) {
      console.error('Error loading video:', error);
    }
  };

  const getSafetyColor = (level) => {
    switch (level) {
      case 'CRITICAL':
        return '#ff3333';
      case 'WARNING':
        return '#ffa500';
      case 'SAFE':
        return '#00cc00';
      default:
        return '#cccccc';
    }
  };

  return (
    <div className="app">
      {/* Navigation Header */}
      <header className="navbar">
        <div className="nav-container">
          <div className="nav-brand">
            <div className="logo-icon">🚦</div>
            <h1>Safety Sentinel</h1>
            <span className="version-badge">v1.1.0</span>
          </div>
          <nav className="nav-links">
            <a href="#" className="nav-link active">Dashboard</a>
            <a href="#" className="nav-link">Docs</a>
            <a href="#" className="nav-link">Settings</a>
          </nav>
          <div className="status-indicator">
            <span className="status-dot" style={{ backgroundColor: stats ? '#00cc00' : '#cccccc' }}></span>
            <span>{stats ? 'Online' : 'Offline'}</span>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h2>Hybrid Deep-Classical Near-Miss Detection</h2>
          <p>Real-time safety analysis at urban intersections using AI + Rule-based Fusion</p>
          <button
            className="btn btn-hero"
            onClick={() => setActiveTab('upload')}
          >
            Analyze Video →
          </button>
        </div>
        <div className="hero-visual">
          <div className="animated-circle"></div>
          <div className="animated-circle secondary"></div>
        </div>
      </section>

      {/* Tab Navigation */}
      <div className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📹 Upload & Analyze
        </button>
        {results && (
          <button
            className={`tab-btn ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
            ✅ Analysis Results
          </button>
        )}
      </div>

      {/* Main Content */}
      <div className="main-wrapper">
        {/* Dashboard Tab - Enhanced Signal-based Grid */}
        {activeTab === 'dashboard' && (
          <div className="dashboard-grid-signal">
            {/* Left: Video + Signal Widget */}
            <div className="signal-panel">
              <div className="signal-container">
                <TrafficSignal
                  level={currentSafetyLevel}
                  riskScore={currentRiskScore}
                  explanation={
                    currentSafetyLevel === 'CRITICAL'
                      ? 'Multiple safety violations detected - immediate action required'
                      : currentSafetyLevel === 'WARNING'
                      ? 'Potential safety concern detected - monitor closely'
                      : 'Normal operating conditions'
                  }
                />
              </div>

              {/* Video placeholder area */}
              <div className="video-feed-area">
                <div className="video-placeholder">
                  <p>📹 Live feed (or select from recent)</p>
                  {events.length > 0 && (
                    <button className="btn-small" onClick={() => console.log('Load event')}>
                      Load Recent Event
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Right: Analytics and Events */}
            <div className="analytics-panel">
              {/* System Stats Cards */}
              {stats && (
                <div className="stats-section">
                  <h3>System Statistics</h3>
                  <div className="stats-cards-grid">
                    <div className="stat-card">
                      <div className="stat-number">{stats.videos_processed}</div>
                      <div className="stat-label">Videos Processed</div>
                    </div>
                    <div className="stat-card critical-stat">
                      <div className="stat-number">{stats.critical_events}</div>
                      <div className="stat-label">Critical Events</div>
                    </div>
                    <div className="stat-card warning-stat">
                      <div className="stat-number">{stats.warning_events}</div>
                      <div className="stat-label">Warnings</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-number">{stats.total_events}</div>
                      <div className="stat-label">Total Events</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Risk Over Time Chart */}
              {liveRiskData.length > 0 && (
                <div className="chart-section">
                  <h3>Risk Score Trend</h3>
                  <RiskChart data={liveRiskData} />
                </div>
              )}

              {/* Recent Events Table */}
              {events.length > 0 && (
                <div className="events-section">
                  <h3>Recent Events</h3>
                  <div className="events-table">
                    <div className="table-header">
                      <div className="col-time">Time</div>
                      <div className="col-level">Level</div>
                      <div className="col-score">Risk Score</div>
                      <div className="col-cause">Cause</div>
                    </div>
                    <div className="table-body">
                      {events.slice(0, 10).map((event, idx) => {
                        const levelColor =
                          event.level === 'CRITICAL'
                            ? '#ff3333'
                            : event.level === 'WARNING'
                            ? '#ffa500'
                            : '#00cc00';
                        return (
                          <div key={idx} className="table-row">
                            <div className="col-time">{event.timestamp?.toFixed(2)}s</div>
                            <div className="col-level">
                              <span
                                className="level-badge"
                                style={{ backgroundColor: levelColor }}
                              >
                                {event.level}
                              </span>
                            </div>
                            <div className="col-score">{(event.risk_score * 100).toFixed(1)}%</div>
                            <div className="col-cause">Near-miss detected</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="upload-container">
            <div className="upload-card">
              <div className="upload-icon">📹</div>
              <h2>Upload Video for Analysis</h2>
              <p>Upload traffic surveillance video for hybrid AI near-miss detection</p>

              <div
                className="upload-drop-zone"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="drop-icon">⬆️</div>
                <p>Click to select or drag video here</p>
                <span className="drop-hint">Supported: MP4, AVI, MOV, MKV</span>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                disabled={loading}
                style={{ display: 'none' }}
              />

              {selectedFile && (
                <div className="selected-file">
                  <div className="file-icon">📄</div>
                  <div className="file-info">
                    <p className="file-name">{selectedFile.name}</p>
                    <p className="file-size">
                      ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  </div>
                </div>
              )}

              <button
                onClick={handleUpload}
                disabled={loading || !selectedFile}
                className="btn btn-primary btn-large"
              >
                {loading ? (
                  <>
                    <span className="spinner"></span> Processing...
                  </>
                ) : (
                  'Analyze Video'
                )}
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && results && (
          <div className="results-container">
            <div className="results-grid-new">
              {/* Left: Video */}
              <div className="results-video">
                <div className="video-header">
                  <h3>Annotated Analysis Video</h3>
                  <span className="video-meta">{results.total_frames} frames processed</span>
                </div>
                <div id="videoContainer" className="video-container">
                  <p className="placeholder">Processing video...</p>
                </div>
              </div>

              {/* Right: Detailed Results */}
              <div className="results-details">
                <div className="results-summary">
                  <h3>Safety Summary</h3>
                  <div className="summary-cards">
                    <div className="summary-card">
                      <div className="summary-label">Total Frames</div>
                      <div className="summary-value">{results.total_frames}</div>
                    </div>
                    <div className="summary-card safe">
                      <div className="summary-label">Safe</div>
                      <div className="summary-value">{results.safety_stats?.SAFE || 0}</div>
                    </div>
                    <div className="summary-card warning">
                      <div className="summary-label">Warning</div>
                      <div className="summary-value">{results.safety_stats?.WARNING || 0}</div>
                    </div>
                    <div className="summary-card critical">
                      <div className="summary-label">Critical</div>
                      <div className="summary-value">{results.safety_stats?.CRITICAL || 0}</div>
                    </div>
                  </div>
                </div>

                {/* Factor Breakdown */}
                <FactorBreakdown
                  factors={{
                    'Deep Learning Anomaly': 0.25,
                    'Distance Violation': 0.35,
                    'Speed Risk': 0.20,
                    'Pedestrian Interaction': 0.15,
                    'TTC (Time-to-Collision)': 0.05,
                  }}
                />

                {/* Critical Events */}
                {results.top_events && results.top_events.length > 0 && (
                  <div className="critical-events">
                    <h4>Top Risk Events</h4>
                    <div className="events-stack">
                      {results.top_events.slice(0, 5).map((event, idx) => (
                        <div
                          key={idx}
                          className="event-stack-item"
                          style={{
                            borderLeft: `4px solid ${
                              event.level === 'CRITICAL'
                                ? '#ff3333'
                                : event.level === 'WARNING'
                                ? '#ffa500'
                                : '#00cc00'
                            }`,
                          }}
                        >
                          <div className="event-time">Frame {event.frame_idx}</div>
                          <div className="event-level" style={{
                            color: event.level === 'CRITICAL'
                              ? '#ff3333'
                              : event.level === 'WARNING'
                              ? '#ffa500'
                              : '#00cc00'
                          }}>
                            {event.level}
                          </div>
                          <div className="event-risk">Risk: {(event.risk_score * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Download Buttons */}
                <div className="results-actions">
                  {results.annotated_video_path && (
                    <a
                      href={`${API_BASE_URL}${results.annotated_video_path}`}
                      className="btn btn-secondary btn-small"
                      download
                    >
                      📥 Download Video
                    </a>
                  )}
                  <button
                    className="btn btn-primary btn-small"
                    onClick={() => setActiveTab('upload')}
                  >
                    📹 Analyze Another
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>Safety Sentinel v1.1.0 | Hybrid Deep-Classical AI for Urban Intersection Safety</p>
          <p className="footer-text">
            Combining LSTM temporal anomaly detection with classical rule-based logic for interpretable
            near-miss detection
          </p>
        </div>
      </footer>
    </div>
  );
}
