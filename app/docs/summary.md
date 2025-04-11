# Car Rental API with Chatbot Integration

## 1. Project Overview
**Project Name**: Development of a Mobile Application with Integrated Smart Chatbot using Speech-to-Text for Car Purchase and Rental Consultation.

### Input/Output Flow
- **Input**: Messages, Voice
- **Output**: Messages, Car options with explanations

### Data Source
Car details from: https://github.com/im-dpaul/EDA-Cars-Data/blob/main/cars_data.csv

## 2. Core Features and Actors

### Main Features

#### Chatbot
[Chatbot Architecture Diagram](https://drive.google.com/file/d/1BN1DNkK7RyXTVim3IK-utJNB_gRnilgh/view?usp=sharing)

1. **Car Purchase/Rental Consultation**:
   - Users can ask natural language questions (text or voice)
   - Chatbot responds with car data, specifications, pros/cons, and recommendations

2. **Speech-to-Text**:
   - Converts voice questions to text
   - Supports multiple languages (Vietnamese and English)

3. **Car Search and Filtering**:
   - Filter by price, brand, type, year, condition, purchase/rental
   - Display detailed car information with images
   - Smart search capabilities:
     - Similarity Search for semantic queries
     - Text-to-SQL Search for specific queries

4. **Car Comparison**:
   - Compare 2-3 cars side by side (specs, price, features)

5. **Automated Data Updates**:
   - Crawl data from reputable car websites to keep information current

6. **Consultation History**:
   - Save previous sessions for reference

7. **Smart Recommendations**:
   - Suggest potential cars or follow-up questions based on interaction history

8. **Car Image Display**:
   - Show relevant car images with chatbot responses

### Actors
1. **Individual Users**: People looking to buy or rent cars
2. **System Administrators**: Monitoring data and chatbot performance
3. **AI Chatbot**: Automated agent processing questions and providing answers

## 3. Challenges, Expected Contributions, and Differentiators

### Challenges
1. **Natural Language Processing**:
   - Understanding unstructured questions, especially in Vietnamese
   
2. **Speech-to-Text Accuracy**:
   - Converting Vietnamese speech with regional accents accurately
   
3. **Data Quality**:
   - Standardizing inconsistent data from different sources
   
4. **Application Performance**:
   - Optimizing chatbot, speech-to-text, and data display in Flutter
   
5. **Real-time Data Updates**:
   - Ensuring continuous data updates without disrupting user experience
   
6. **Complex Data Search**:
   - Processing both semantic and specific queries efficiently
   
7. **Cross-platform Integration**:
   - Ensuring smooth data search on both iOS and Android

### Expected Contributions
1. **Comprehensive Solution**:
   - Integrating chatbot, speech-to-text, and crawled data in a Flutter app
   
2. **Automated Consultation**:
   - Reducing dependency on traditional consultants
   
3. **Advanced Vietnamese Language Support**:
   - Focusing on Vietnamese language processing
   
4. **Rich and Updated Data**:
   - Using crawled data instead of static databases
   
5. **Intelligent Search**:
   - Combining Similarity Search and Text-to-SQL Search

## 4. Data Search Implementation Method

### Search Methods
- **Similarity Search**:
  - Vector-based similarity search using Sentence Transformers and Pinecone/Faiss
  
- **Text-to-SQL Search**:
  - Converting specific queries to SQL using NLP models or fixed templates

### Technologies
- **Backend**: Python (FastAPI), MongoDB, ChromaDB
- **Frontend**: Flutter with HTTP/GraphQL API calls
- **AI**: Speech-to-text, RAG, Function calling (Agent Building)

### Development Priorities
1. Backend development first:
   - Build RAG system with vector database
   - User database (chat history, information)
   - Speech-to-text (Live transcribe)
   
2. API endpoints:
   - Login/Register
   - Chat (images and text)

### Running the Backend
