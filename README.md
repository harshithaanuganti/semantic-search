# Semantic-Search for Hotel reviews in Phuket

This app will let the user search into the reviews of various hotels in Phuket Island, Thailand.
https://harshithaanuganti-semantic-search-app-40yrps.streamlit.app/

The user can enter the text in the text box(as shown) and press enter to see the top 5 hotels whose reviews match with the entered text.

![Search](https://user-images.githubusercontent.com/74675390/206099137-39d33613-ae44-4309-bf43-ee6586ea9f68.png)


As shown, the app shows a summarized review of the hotels along with the wordcloud of the reviews to give the user more understanding about the experience of the previous hotel guests.

![Output](https://user-images.githubusercontent.com/74675390/206099202-c19f9a41-f2ad-4d4f-bf3f-36a23dc039b3.png)


Here, after the basic pre-processing of the data, the reviews of each hotel have been grouped together and summarized using *gensim summarizer*. A pre-trained *Sentence transformer* model is used to create corpus and query(user-input) embeddings, which are then compared by calculating their cosine similarity.
The application is deployed on *streamlit*. To reduce the runtime, the dataframe, corpus, corpus embeddings and the model have been exported as pickle files to import into the streamlit application.
*Wordcloud* has been used to give the user more understanding on how the hotel reviews look like.
    
        
