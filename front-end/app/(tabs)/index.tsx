import React, { useState, useCallback } from 'react';
import { View, StyleSheet } from 'react-native';
import { GiftedChat } from 'react-native-gifted-chat';
import axios from 'axios';

export default function Chatbot() {
  const [messages, setMessages] = useState([]);

  // Handle sending messages
  const onSend = useCallback((newMessages = []) => {
    setMessages((previousMessages) =>
      GiftedChat.append(previousMessages, newMessages)
    );

    const userMessage = newMessages[0];

    // Example API call to fetch a response
    axios
      .post('https://api.example.com/chat', { message: userMessage })
      .then((response) => {
        const botMessage = {
          _id: Math.random().toString(36).substring(7), // Unique ID for the message
          text: response.data.reply, // Response text from the API
          createdAt: new Date(),
          user: {
            _id: 2,
            name: 'Chatbot',
          },
        };
      })
      .catch((error) => {
        console.error(error);
        const errorMessage = {
          _id: Math.random().toString(36).substring(7),
          text: 'Oops! Something went wrong. Please try again later.',
          createdAt: new Date(),
          user: {
            _id: 2,
            name: 'Chatbot',
          },
        };
      });
  }, []);

  return (
    <View style={styles.container}>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});
