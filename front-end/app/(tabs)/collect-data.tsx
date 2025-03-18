import React, { useState, useRef } from "react";
import { View, Text, TextInput, FlatList, TouchableOpacity } from "react-native";
import { Picker } from "@react-native-picker/picker";
import Toast from "react-native-toast-message";
import styles from "../styles/chat.styles"; // Reuse your existing styles or create new ones

// URL of your FastAPI server (adjust port if needed)
const API_URL = "http://127.0.0.1:8200";

interface Message {
  text: string;
  user?: boolean;
  error?: boolean;
}

const CollectData = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  // Dropdown state: selected command type
  const [selectedCommand, setSelectedCommand] = useState("forward");
  // Additional text input for command details (e.g., parameters)
  const [commandDetails, setCommandDetails] = useState("");
  const [loading, setLoading] = useState(false);

  const flatListRef = useRef<FlatList>(null);

  const sendCommand = async () => {
    if (!selectedCommand.trim()) return;
    setLoading(true);

    // Log the user's selection and details
    setMessages((prev) => [
      ...prev,
      { text: `Selected Command: ${selectedCommand}`, user: true },
      { text: `Details: ${commandDetails}`, user: true },
    ]);

    // Construct payload based on the selected command and details
    // Assuming the backend expects { action: string, parameters: object }
    let parsedDetails = {};
    if (commandDetails.trim()) {
      try {
        parsedDetails = JSON.parse(commandDetails);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { text: "Details must be valid JSON (e.g., {\"distance\": 50}).", user: false, error: true },
        ]);
        setLoading(false);
        return;
      }
    }

    const payload = {
      action: selectedCommand,
      parameters: parsedDetails,
    };

    try {
      const response = await fetch(`${API_URL}/user-data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Error storing user data.");
      }
      const result = await response.json();
      setMessages((prev) => [
        ...prev,
        { text: `Data saved: ${result.filename}`, user: false },
      ]);
    } catch (error: any) {
      console.error("Error sending command:", error);
      setMessages((prev) => [
        ...prev,
        { text: error.message || "Error sending command.", user: false, error: true },
      ]);
    }

    setSelectedCommand("forward");
    setCommandDetails("");
    setLoading(false);
  };

  const renderMessage = ({ item }: { item: Message }) => (
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
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
      />

      {/* Dropdown for selecting command */}
      <View style={styles.inputContainer}>
        <Picker
          selectedValue={selectedCommand}
          onValueChange={(itemValue) => setSelectedCommand(itemValue)}
          style={styles.picker}
        >
          <Picker.Item label="Forward" value="forward" />
          <Picker.Item label="Back" value="back" />
          <Picker.Item label="Left" value="left" />
          <Picker.Item label="Right" value="right" />
          <Picker.Item label="Stop" value="stop" />
          {/* Add other command options as needed */}
        </Picker>
      </View>

      {/* TextInput for command details */}
      <View style={styles.inputContainer}>
        <TextInput
          placeholder='Enter command details in JSON (e.g., {"distance": 50})'
          onChangeText={setCommandDetails}
          value={commandDetails}
          style={styles.input}
          placeholderTextColor="#fff"
          editable={!loading}
        />
      </View>

      <TouchableOpacity style={styles.sendButton} onPress={sendCommand} disabled={loading}>
        <Text style={styles.sendButtonText}>{loading ? "Sending..." : "Send"}</Text>
      </TouchableOpacity>
      <Toast />
    </View>
  );
};

export default CollectData;
