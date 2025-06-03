import React, { useState, useEffect } from 'react';
import './DevLogger.css';

interface LogMessage {
  timestamp: string;
  message: string;
  type: 'info' | 'analysis' | 'decision';
}

export function DevLogger() {
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [isMinimized, setIsMinimized] = useState(false);

  useEffect(() => {
    // Override console.log to capture messages
    const originalLog = console.log;
    
    console.log = (...args) => {
      // Call original console.log
      originalLog(...args);
      
      // Capture AI-related logs
      const message = args.join(' ');
      if (message.includes('COMPUTER TURN') || 
          message.includes('Computer') || 
          message.includes('analysis:') ||
          message.includes('[Neural Network AI]')) {
        
        let type: 'info' | 'analysis' | 'decision' = 'info';
        if (message.includes('analysis:')) type = 'analysis';
        if (message.includes('COMPUTER TURN')) type = 'decision';
        
        setLogs(prev => [...prev.slice(-50), {
          timestamp: new Date().toLocaleTimeString(),
          message: message,
          type: type
        }]);
      }
    };
    
    return () => {
      console.log = originalLog;
    };
  }, []);

  const clearLogs = () => setLogs([]);

  if (isMinimized) {
    return (
      <div className="dev-logger minimized" onClick={() => setIsMinimized(false)}>
        <span>📋 Show AI Logs</span>
      </div>
    );
  }

  return (
    <div className="dev-logger">
      <div className="dev-logger-header">
        <h3>AI Decision Log</h3>
        <div className="dev-logger-controls">
          <button onClick={clearLogs}>Clear</button>
          <button onClick={() => setIsMinimized(true)}>_</button>
        </div>
      </div>
      <div className="dev-logger-content">
        {logs.length === 0 ? (
          <div className="no-logs">Waiting for AI moves...</div>
        ) : (
          logs.map((log, index) => (
            <div key={index} className={`log-entry log-${log.type}`}>
              <span className="log-time">{log.timestamp}</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}