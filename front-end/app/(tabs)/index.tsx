import React, { useState, useCallback } from 'react';
import { View, StyleSheet } from 'react-native';
import { GiftedChat } from 'react-native-gifted-chat';
import axios from 'axios';

export default function Chatbot() {
  const [messages, setMessages] = useState([]);

  // Initialize with a default message from the bot
  React.useEffect(() => {
    setMessages([
      {
        _id: 1,
        text: 'Hello! How can I assist you today?',
        createdAt: new Date(),
        user: {
          _id: 2,
          name: 'Chatbot',
        },
      },
    ]);
  }, []);

  // Handle sending messages
  const onSend = useCallback((newMessages = []) => {
    setMessages((previousMessages) =>
      GiftedChat.append(previousMessages, newMessages)
    );

    const userMessage = newMessages[0].text;

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

        setMessages((previousMessages) =>
          GiftedChat.append(previousMessages, [botMessage])
        );
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
        setMessages((previousMessages) =>
          GiftedChat.append(previousMessages, [errorMessage])
        );
      });
  }, []);

  return (
    <View style={styles.container}>
      <GiftedChat
        messages={messages}
        onSend={(messages) => onSend(messages)}
        user={{
          _id: 1, // User ID of the app user
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});
