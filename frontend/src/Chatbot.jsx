import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./Chatbot.css";

const WS_URL = "ws://localhost:8000/ws/chat/";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi! I am your AI Assistant for Landlords. Ask me about rent predictions or property advice!" },
  ]);
  const [input, setInput] = useState("");
  const ws = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    ws.current = new window.WebSocket(WS_URL);
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "property_response") {
        setMessages((msgs) => [
          ...msgs,
          {
            sender: "bot",
            text: data.explanation || `\n🎯 Predicted Rent: £${data.predicted_rent}\n💡 Rent Range: £${data.lower_rent}–£${data.upper_rent}\n🔒 Confidence: ${data.confidence_percentage}%\n📝 ${data.explanation}`,
          },
        ]);
      } else if (data.type === "bot_response") {
        setMessages((msgs) => [
          ...msgs,
          { sender: "bot", text: data.message }
        ]);
      } else if (data.type === "echo") {
        setMessages((msgs) => [...msgs, { sender: "bot", text: data.message }]);
      } else if (data.type === "error") {
        setMessages((msgs) => [...msgs, { sender: "bot", text: `Error: ${data.error}` }]);
      }
    };
    ws.current.onclose = () => {
    //   setMessages((msgs) => [...msgs, { sender: "bot", text: "Connection closed." }]);
    };
    return () => ws.current && ws.current.close();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setMessages((msgs) => [...msgs, { sender: "user", text: input }]);
    if (input.trim().toLowerCase() === "/predict") {
      // Send property data for prediction
      ws.current.send(
        JSON.stringify({
          type: "property",
          input_data: {
            address: 1308,
            subdistrict_code: 182,
            BEDROOMS: 2.0,
            BATHROOMS: 1.0,
            SIZE: 700.0,
            "PROPERTY TYPE": 9,
            avg_distance_to_nearest_station: 0.4,
            nearest_station_count: 3.0
          }
        })
      );
    } else {
      // Send as normal text
      ws.current.send(
        JSON.stringify({
          type: "text",
          message: input,
        })
      );
    }
    setInput("");
  };

  return (
    <div className="chatbot-container" style={{ minHeight: '85vh', height: '80dvh', display: 'flex', flexDirection: 'column' }}>
      <div className="chatbot-header">AI for Landlords</div>
      <div className="chatbot-messages" style={{ flex: 1, overflowY: 'auto' }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`chatbot-message chatbot-message-${msg.sender}`}
            style={{ textAlign: 'left' }}
          >
            {msg.sender === "bot" ? (
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            ) : (
              msg.text.split("\n").map((line, i) => <div key={i}>{line}</div>)
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      <form className="chatbot-input-area" onSubmit={handleSend}>
        <input
          className="chatbot-input"
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button className="chatbot-send-btn" type="submit">
          ➤
        </button>
      </form>
    </div>
  );
}
