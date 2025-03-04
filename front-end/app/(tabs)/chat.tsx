import React, { useState, useEffect } from "react";
import * as GoogleGenerativeAI from "@google/generative-ai";
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
} from "react-native";
import Toast from "react-native-toast-message";
import styles from "../styles/chat.styles";  

const Chat = () => {
  const [messages, setMessages] = useState([
    { text: "", user: false }
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);

  const API_KEY = ""; 

  useEffect(() => {
    const startChat = async () => {
      if (!API_KEY) {
        setMessages([{ text: "⚠️ API Key Missing! Please add your key to continue.", user: false, error: true }]);
        return;
      }

      try {
        const genAI = new GoogleGenerativeAI.GoogleGenerativeAI(API_KEY);
        const model = genAI.getGenerativeModel({ model: "gemini-pro" });
        const prompt = "Hello! ";
        const result = await model.generateContent(prompt);
        const response = result.response;
        const text = response.text();

        setMessages([{ text, user: false }]);
      } catch (error) {
        setMessages([{ text: "⚠️ Error connecting to AI model. Please try again later.", user: false, error: true }]);
      }
    };

    startChat();
  }, []);

  const sendMessage = async () => {
    if (!userInput.trim()) return;

    setLoading(true);
    const userMessage = { text: userInput, user: true };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setUserInput("");

    try {
      const genAI = new GoogleGenerativeAI.GoogleGenerativeAI(API_KEY);
      const model = genAI.getGenerativeModel({ model: "gemini-pro" });
      const result = await model.generateContent(userMessage.text);
      const response = result.response;
      const text = response.text();

      setMessages((prevMessages) => [...prevMessages, { text, user: false }]);
      setLoading(false);
    } catch (error) {
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "⚠️ AI failed to generate a response. Please try again.", user: false, error: true }
      ]);
      setLoading(false);
    }
  };

  const renderMessage = ({ item }) => (
    <View style={[styles.messageContainer, item.user ? styles.userMessageContainer : styles.aiMessageContainer]}>
      <Text style={[styles.messageText, item.error ? styles.errorMessage : item.user ? styles.userMessage : styles.aiMessage]}>
        {item.text}
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(_, index) => index.toString()}
        inverted
      />
      <View style={styles.inputContainer}>
        <TextInput
          placeholder="Type a message"
          onChangeText={setUserInput}
          value={userInput}
          onSubmitEditing={sendMessage}
          style={styles.input}
          placeholderTextColor="#fff"
        />
        <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </View>
      <Toast />
    </View>
  );
};

export default Chat;
