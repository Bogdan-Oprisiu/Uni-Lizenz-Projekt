import { StyleSheet } from "react-native";

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

export default styles;
