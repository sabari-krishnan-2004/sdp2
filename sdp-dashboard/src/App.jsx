import React, { useState, useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';
import './index.css';

// --- COMPONENTS ---

const Card = ({ title, value, subtitle, icon }) => (
    <div className="card glass">
        <div className="card-header">
            <h3 className="card-title">{title}</h3>
            <span className="card-icon">{icon}</span>
        </div>
        <div className="card-value">{value}</div>
        {subtitle && <div className="card-subtitle">{subtitle}</div>}
    </div>
);

const AccuracyChart = ({ history }) => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);

    useEffect(() => {
        if (chartInstance.current) {
            chartInstance.current.destroy();
        }

        if (!history || history.length === 0) return;

        const ctx = chartRef.current.getContext('2d');
        
        // Prepare data
        const labels = history.map(item => `Round ${item.round}`);
        const data = history.map(item => item.accuracy * 100);

        chartInstance.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Global Accuracy (%)',
                    data: data,
                    borderColor: '#8b5cf6', // Violet
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#c4b5fd',
                    pointBorderColor: '#8b5cf6',
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e2e8f0', font: { family: 'Inter', size: 14 } }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#cbd5e1',
                        padding: 12,
                        cornerRadius: 8
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    }
                },
                animation: {
                    duration: 1500,
                    easing: 'easeOutQuart'
                }
            }
        });

        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
        };
    }, [history]);

    return (
        <div className="chart-container glass">
            <h3 className="section-title">Model Performance</h3>
            <div style={{ position: 'relative', height: '300px', width: '100%' }}>
                {history && history.length > 0 ? (
                    <canvas ref={chartRef}></canvas>
                ) : (
                    <div className="empty-state">No training history available yet.</div>
                )}
            </div>
        </div>
    );
};

const DownloadSection = ({ versionInfo }) => {
    const apiBase = "http://127.0.0.1:8000";
    
    return (
        <div className="download-section glass">
            <h3 className="section-title">Global Model Assets</h3>
            <p className="section-desc">Download the latest aggregated models for inference or further training.</p>
            
            <div className="download-cards">
                <div className="download-card">
                    <div className="dl-info">
                        <h4>TFLite Model</h4>
                        <span>Optimized for Mobile (Flutter/Android/iOS)</span>
                        <span className="dl-version">Version: {versionInfo?.version || 0}</span>
                    </div>
                    <a href={`${apiBase}/model/tflite`} className="btn btn-primary" download>
                        Download .tflite
                    </a>
                </div>
                
                <div className="download-card">
                    <div className="dl-info">
                        <h4>Keras Model (H5)</h4>
                        <span>Full backend model weights</span>
                        <span className="dl-version">Version: {versionInfo?.version || 0}</span>
                    </div>
                    <a href={`${apiBase}/model/latest`} className="btn btn-secondary" download>
                        Download .h5
                    </a>
                </div>
            </div>
        </div>
    );
};

// --- MAIN APP ---

const App = () => {
    const [info, setInfo] = useState(null);
    const [history, setHistory] = useState([]);
    const [versionInfo, setVersionInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const API_BASE = "http://127.0.0.1:8000";

    const fetchData = async () => {
        try {
            const [infoRes, metricsRes, versionRes] = await Promise.all([
                fetch(`${API_BASE}/info`).then(r => r.json()),
                fetch(`${API_BASE}/metrics`).then(r => r.json()),
                fetch(`${API_BASE}/model/version`).then(r => r.json())
            ]);

            setInfo(infoRes);
            setHistory(metricsRes.history || []);
            setVersionInfo(versionRes);
            setError(null);
        } catch (err) {
            console.error("Failed to fetch data:", err);
            setError("Could not connect to API Server. Ensure it is running on port 8000.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        // Refresh every 5 seconds
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="loading-screen">
                <div className="spinner"></div>
                <h2>Loading Dashboard...</h2>
            </div>
        );
    }

    return (
        <div className="app-container">
            <header className="app-header glass">
                <div className="header-content">
                    <div className="logo-area">
                        <div className="logo-icon">💠</div>
                        <h1>Federated Emotion Network</h1>
                    </div>
                    <div className={`status-badge ${error ? 'offline' : 'online'}`}>
                        <span className="status-dot"></span>
                        {error ? 'API Offline' : 'API Online'}
                    </div>
                </div>
            </header>

            <main className="app-main">
                {error && (
                    <div className="error-banner glass">
                        ⚠️ {error}
                    </div>
                )}

                <div className="dashboard-grid">
                    <Card 
                        title="Current Round" 
                        value={info?.current_round || 0} 
                        subtitle="Federated Learning Cycles"
                        icon="🔄"
                    />
                    <Card 
                        title="Global Accuracy" 
                        value={`${((info?.latest_accuracy || 0) * 100).toFixed(1)}%`} 
                        subtitle="Across all edge clients"
                        icon="🎯"
                    />
                    <Card 
                        title="Total Participants" 
                        value={info?.total_clients || 0} 
                        subtitle="Devices contributed"
                        icon="📱"
                    />
                </div>

                <div className="main-content-grid">
                    <AccuracyChart history={history} />
                    <DownloadSection versionInfo={versionInfo} />
                </div>
            </main>
        </div>
    );
};

export default App;
