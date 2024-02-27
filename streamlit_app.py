#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    
    st.title('Pixel Art Classification')
    st.subheader('by Fatima Grace Apinan|| BSCS-3B || WVSU CICT')
    
    st.write('Dataset description:')

    st.write('Number of features: 64')
    text = """This code implements a logistic regression model to recognize pixel art images.
    The dataset used consists of pixelated images represented as binary values (1 or 0) in an 8x8 grid, with each pixel indicating the presence or absence of a feature.
    The target variable represents the class of the image, indicating the specific category or label associated with each image.
    Features: """
    st.write(text)
    st.write('Feature Representation: Binary values representing the 8x8 pixels of each image.')
    st.write('Target Variable: Categorical variable representing the class of the image.')

    # display choice of classifier
    clf = BernoulliNB() 
    options = ['Naive Bayes', 'Logistic Regression']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option=='Logistic Regression':
        clf = LogisticRegression(C=100, max_iter=100, multi_class='auto',
            penalty='l2', random_state=42, solver='lbfgs',
            verbose=0, warm_start=False)
    else:
        clf = BernoulliNB()

    if st.button('Start'):
        df = pd.read_csv('Arendain_Smiley.csv', header=None)
        # st.dataframe(df, use_container_width=True)  
        
        # display the dataset
        st.header("Dataset")
        st.dataframe(df, use_container_width=True) 

        #load the data and the labels
        X = df.values[:,0:-1]
        y = df.values[:,-1]    

        st.header('Images')
        # display the images 
        fig, axs = plt.subplots(4, 10, figsize=(20, 8))

        # Iterate over the images and labels
        for index, (image, label) in enumerate(zip(X, y)):
            # Get the corresponding axes object
            ax = axs.flat[index]

            # Display the image
            ax.imshow(np.reshape(image, (8, 8)), cmap='binary')

        # Add the title
        ax.set_title(f'Training: {label}', fontsize=10)

        # Tighten layout to avoid overlapping
        plt.tight_layout()
        st.pyplot(fig)
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)

        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)

        st.header('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.header('Classification Report')
        # Test the classifier on the testing set
        st.text(classification_report(y_test, y_test_pred))
    
#run the app
if __name__ == "__main__":
    app()
