import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx

# Load the dataset
def load_data():
    df = pd.read_csv('course_info.csv')

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
def get_recommendation(title, cosine_sim, df, num_of_rec=10):
    # indices of the course
    course_indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    # Index of course
    idx = course_indices[title]

    # Look into the cosine matr for that index
    sim_scores =list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['title','similarity_score','course_url','price','num_subscribers']]
    return final_recommended_courses.head(num_of_rec)


# Search For Course 
@st.cache
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df


def main():
    # Page setting
    st.set_page_config(layout="wide")
    st.title('Udemy Course Recommendation System')

    menu = ["Home","Recommend","About"]
    for name in menu:
        choice = st.sidebar.button(name)

    course_data = load_data()

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(course_data.head(10))

    elif choice == "Recommend":
        st.subheader("Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(course_data['title'])
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number",4,30,7)

        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    with st.beta_expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)


                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_avg_rating = row[1][6]
                        rec_url = row[1][17]
                        rec_price = row[1][3]
                        rec_num_sub = row[1][4]

                    
                        stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_url,rec_num_sub),height=350)

                except:
                    results= "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term, course_data)
                    st.dataframe(result_df)

    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
    main()
