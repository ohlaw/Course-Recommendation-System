import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx

# Load the dataset

course_info = ('course_info.csv')

def load_data(data):
    df = pd.read_csv(data)

    df.drop(['is_paid', 'headline', 'num_comments', 'last_update_date', 'instructor_url'], axis=1)
	
    # Convert date columns to datetime data type
    df['published_time'] = pd.to_datetime(df['published_time'])

    # Replace missing dates in topic, and instructor_name with 'unknown'
    df['topic'].fillna('unknown', inplace=True)
    df['instructor_name'].fillna('unknown', inplace=True)

    # Convert columns to integers
    numeric_columns = ['id', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_length_min']
    df[numeric_columns] = df[numeric_columns].astype(int)

    # Clean text:stopwords,special characters
    df['title'] = df['title'].apply(nfx.remove_stopwords)

    # Clean Text:stopwords,special charac
    df['title'] = df['title'].apply(nfx.remove_special_characters)

    return df


def vectorize_text_to_cosine_mat(df):
    # vectorize the text and compute the cosine similarity
    count_vect = CountVectorizer()
    cv_matrix = count_vect.fit_transform(df)
    cosine_sim = cosine_similarity(cv_matrix)
    return cosine_sim
    

# Recommendation System
@st.cache
def get_recommendation(title, df, num_of_rec):
    # indices of the course
    course_indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    # Index of course
    idx = course_indices[title]

    # Look into the cosine matrix for that index
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['title','similarity_score','course_url','price','num_subscribers']]
    final_recommended_courses = final_recommended_courses.head(num_of_rec)
    return final_recommended_courses


@st.cache_data
# Search For Course 
def search_term_if_not_found(term, df):
    result_df = df[df['title'].str.contains(term)]
    return result_df

def main():
    # Page setting
    st.set_page_config(layout="wide")
    st.title('Udemy Course Recommendation System')

    menu = ['Home', 'Recommend', 'About']
    choice = st.sidebar.selectbox("Menu", menu)

    course_data = load_data(course_info)

    if choice == 'Home':
        st.subheader("Home")
        st.write(course_data.head(10))        
        

    elif choice == 'Recommend':
        st.subheader("Get Your Course Recommendation")
        search_term = st.text_input("Enter a course title:")
        num_of_rec = st.slider("Number of recommendations:", 1, 10, 5)
        if st.button("Get Recommendations"):
            if search_term is not None:
                try:
                    with st.spinner("Fetching recommendations..."):
                        recommendations = get_recommendation(search_term, course_data, num_of_rec)
                        st.write(recommendations)
                    
                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term, course_data)
                    st.dataframe(result_df.head(20))

    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")

if __name__ == '__main__':
    main()
