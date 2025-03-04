import React, { useState, useEffect, useRef } from "react";
import { View, Text, TextInput, FlatList, TouchableOpacity } from "react-native";
import Toast from "react-native-toast-message";
import styles from "../styles/chat.styles"; // <-- Import styles from a separate file

// URL of your FastAPI server
const API_URL = "http://127.0.0.1:8000";

const Chat = () => {
  // Store game state from the API (optional usage)
  const [gameState, setGameState] = useState(null);
  // We use messages for a chat-style log
  const [messages, setMessages] = useState<{ text: string; user: boolean; error?: boolean }[]>([]);
  const [commandInput, setCommandInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Reference for the FlatList to auto-scroll to bottom
  const flatListRef = useRef<FlatList>(null);

  // When component mounts, fetch initial game state
  useEffect(() => {
    fetch(`${API_URL}/state`)
      .then((response) => response.json())
      .then((data) => {
        setGameState(data);
        setMessages([{ text: `Game started. Score: ${data.score}`, user: false }]);
      })
      .catch((error) => {
        console.error("Error fetching game state:", error);
        setMessages([{ text: "Error fetching game state", user: false, error: true }]);
      });
  }, []);

  const sendCommand = async () => {
    if (!commandInput.trim()) return;
    setLoading(true);

    // Log user's command
    setMessages((prev) => [...prev, { text: `Command: ${commandInput}`, user: true }]);

    try {
      const response = await fetch(`${API_URL}/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: commandInput }),
      });

      if (!response.ok) {
        throw new Error("Game API error");
      }
      const updatedState = await response.json();
      setGameState(updatedState);

      // Optionally log updated score, sensors, etc.
      setMessages((prev) => [
        ...prev,
        { text: `Score: ${updatedState.score}`, user: false },
        { text: `Sensors: ${JSON.stringify(updatedState.sensors)}`, user: false },
      ]);
    } catch (error) {
      console.error("Error sending command:", error);
      setMessages((prev) => [
        ...prev,
        { text: "Error sending command. Please try again.", user: false, error: true },
      ]);
    }

    setCommandInput("");
    setLoading(false);
  };

  const renderMessage = ({ item }: { item: { text: string; user: boolean; error?: boolean } }) => (
    <View
      style={[
        styles.messageContainer,
        item.user ? styles.userMessageContainer : styles.aiMessageContainer,
      ]}
    >
      <Text
        style={[
          styles.messageText,
          item.error ? styles.errorMessage : item.user ? styles.userMessage : styles.aiMessage,
        ]}
      >
        {item.text}
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(_, index) => index.toString()}
        // Scroll to bottom whenever content changes
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
      />

      <View style={styles.inputContainer}>
        <TextInput
          placeholder="Enter command (e.g., 'forward 50')"
          onChangeText={setCommandInput}
          value={commandInput}
          onSubmitEditing={sendCommand}
          style={styles.input}
          placeholderTextColor="#fff"
        />
        <TouchableOpacity style={styles.sendButton} onPress={sendCommand}>
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </View>
      <Toast />
    </View>
  );
};

export default Chat;
