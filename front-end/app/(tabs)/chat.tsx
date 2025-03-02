import React, { useState, useEffect } from "react";
import * as GoogleGenerativeAI from "@google/generative-ai";
import {
  View,
  Text,
  TextInput,
  FlatList,
  StyleSheet,
  TouchableOpacity,
} from "react-native";
import Toast from "react-native-toast-message";

const Chat = () => {
  const [messages, setMessages] = useState<{ text: string; user: boolean; error?: boolean }[]>([]);
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

  const renderMessage = ({ item }: { item: { text: string; user: boolean; error?: boolean } }) => (
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
        keyExtractor={(item, index) => index.toString()}
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

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 10,
    paddingTop: 50,
    backgroundColor: "#1e1d31", // Dark greyish purple
  },
  messageContainer: {
    padding: 12,
    marginVertical: 5,
    borderRadius: 12,
    maxWidth: "75%",
  },
  userMessageContainer: {
    backgroundColor: "#5a4fcf",
    alignSelf: "flex-start", // Align user messages to the left
  },
  aiMessageContainer: {
    backgroundColor: "#2f2e41",
    alignSelf: "flex-end", // Align AI responses to the right
  },
  messageText: {
    fontSize: 16,
  },
  userMessage: {
    color: "#fff",
    textAlign: "left",
  },
  aiMessage: {
    color: "#ddd",
    textAlign: "right",
  },
  errorMessage: {
    color: "#ff6961", // Red for error messages
    fontWeight: "bold",
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "center",
    padding: 10,
    borderRadius: 15,
    backgroundColor: "#2f2e41",
  },
  input: {
    flex: 1,
    padding: 10,
    backgroundColor: "#1e1d31",
    borderRadius: 10,
    height: 50,
    color: "white",
  },
  sendButton: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    backgroundColor: "#5a4fcf",
    borderRadius: 10,
    marginLeft: 5,
  },
  sendButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
  },
});

export default Chat;