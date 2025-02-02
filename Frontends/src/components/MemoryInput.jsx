// // src/components/MemoryInput.js
// import React, { useState } from "react";
// import axios from "axios";
// import "./MemoryInput.css";

// const MemoryInput = () => {
//   const [memory, setMemory] = useState("");
//   const [status, setStatus] = useState("");

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!memory.trim()) return;

//     setStatus("Saving...");
//     try {
//       const response = await axios.get(
//         "http://localhost:5000/postMemory?text=" + memory
//       );
//       console.log(response);
//       setStatus("Memory saved successfully!");
//       setMemory("");
//     } catch (error) {
//       console.error("Error saving memory:", error);
//       setStatus("Error saving memory. Please try again.");
//     }
//   };

//   return (
//     <div className="memory-input">
//       <h2>Add a New Memory</h2>
//       <form onSubmit={handleSubmit}>
//         <textarea
//           value={memory}
//           onChange={(e) => setMemory(e.target.value)}
//           placeholder="Enter your memory..."
//           rows="4"
//         />
//         <button type="submit">Save Memory</button>
//       </form>
//       {status && <p className="status">{status}</p>}
//     </div>
//   );
// };

// export default MemoryInput;


import React, { useState } from 'react';
import './MemoryInput.css';

const MemoryInput = () => {
  const [memory, setMemory] = useState("");
  const [status, setStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [characterCount, setCharacterCount] = useState(0);

  const handleInput = (e) => {
    const text = e.target.value;
    if (text.length <= 500) {
      setMemory(text);
      setCharacterCount(text.length);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!memory.trim()) return;

    setIsLoading(true);
    setStatus("saving");
    
    try {
      const response = await fetch(
        `http://localhost:8080/postMemory?text=${encodeURIComponent(memory)}`
      );
      
      if (!response.ok) throw new Error('Failed to save');
      
      setStatus("success");
      setMemory("");
      setCharacterCount(0);
      setTimeout(() => setStatus(""), 3000);
    } catch (error) {
      console.error("Error saving memory:", error);
      setStatus("error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="journal-container">
      <div className="paper-effect">
        <div className="memory-header">
          <h1>Memory Journal</h1>
          <p className="date">{new Date().toLocaleDateString('en-US', { 
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
          })}</p>
        </div>

        <form onSubmit={handleSubmit} className="journal-form">
          <div className="textarea-wrapper">
            <div className="lines"></div>
            <textarea
              value={memory}
              onChange={handleInput}
              placeholder="Dear Journal..."
              className="journal-textarea"
              disabled={isLoading}
            />
            <div className="character-counter">
              <div className="counter-bubble">
                {characterCount}/500
              </div>
            </div>
          </div>

          <div className="button-container">
            <button
              type="submit"
              disabled={!memory.trim() || isLoading}
              className="save-button"
            >
              {isLoading ? (
                <div className="button-content">
                  <div className="ink-drop"></div>
                  <span>Preserving memory...</span>
                </div>
              ) : (
                <div className="button-content">
                  <span>✍️ Preserve this memory</span>
                </div>
              )}
            </button>
          </div>

          {status && (
            <div className={`status-popup ${status}`}>
              {status === 'success' && (
                <div className="status-content">
                  <div className="status-icon">✨</div>
                  <p>Memory beautifully preserved</p>
                </div>
              )}
              {status === 'error' && (
                <div className="status-content">
                  <div className="status-icon">❌</div>
                  <p>Couldn't save your memory. Let's try again?</p>
                </div>
              )}
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default MemoryInput;



