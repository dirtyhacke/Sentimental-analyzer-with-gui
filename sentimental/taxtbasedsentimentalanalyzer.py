import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import customtkinter as ctk
from tkinter import filedialog, messagebox
from time import sleep
# Function to preprocess text
# Initialize global variables
nb_model = None
vectorizer = None
accuracy, report, confusion = None, None, None

# Function to preprocess text
def preprocess_text(text):
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for',
        'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or',
        'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they',
        'this', 'to', 'was', 'will', 'with'
    }
    tokens = text.split()
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Function to load and preprocess dataset
def load_data():
    global accuracy, report, confusion, vectorizer, nb_model
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            messagebox.showinfo("SUCCESS", "UPLOADED ✅")
            sleep(2)
            messagebox.showinfo("WAITING", "Please wait. Machine is learning...")

            if 'Review' not in df.columns or 'Review' not in df.columns:
                messagebox.showerror("Error", "Dataset must have 'Review' and 'Sentiment' columns.")
                return

            df['Review'] = df['Review'].fillna('').apply(preprocess_text)
            X = df['Review']
            y = df['Review']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            nb_model = MultinomialNB()
            nb_model.fit(X_train_tfidf, y_train)
            y_pred = nb_model.predict(X_test_tfidf)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            messagebox.showinfo("SUCCESS", "Machine learned successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def find_acc():
    results_textbox.insert("1.0", f"Accuracy: {accuracy}\n")

def find_report():
    results_textbox.insert("1.0", f"Classification Report:\n{report}\n")

def find_confusion():
    results_textbox.insert("1.0", f"Confusion Matrix:\n{confusion}\n")

def clear():
    results_textbox.delete("1.0", "end")

def get_sentiment():
    global nb_model, vectorizer
    if nb_model is None or vectorizer is None:
        messagebox.showerror("Error", "Please train your model first!")
        return

    user_input = predict.get()
    if not user_input:
        messagebox.showerror("Error", "Please enter some text for prediction!")
        return

    try:
        processed_input = preprocess_text(user_input)
        input_tfidf = vectorizer.transform([processed_input])
        prediction = nb_model.predict(input_tfidf)

        predict_result.delete("1.0", "end")
        predict_result.insert("1.0", f"The sentiment is: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
# Create GUI with customtkinter
ctk.set_appearance_mode("Dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

app = ctk.CTk()
app.title("Text Classification with Naive Bayes")
app.geometry("1020x1080")


# Heading label
heading_label = ctk.CTkLabel(app, text="Text Classification with Naive Bayes", font=("Arial", 20, "bold"))
heading_label.pack(pady=10)

    
# Load button
load_button = ctk.CTkButton(app, text="Upload Dataset⬆", command=load_data,hover_color="green"
,fg_color="blue",border_color="white",border_width=0,

)
load_button.place(x=50,y=780)
    
accuracy_button=ctk.CTkButton(app,text="Accurracy🎯",command=find_acc,hover_color="green"
,fg_color="blue" ,border_color="white",border_width=0         
                              )
accuracy_button.place(x=200,y=780)
    
report_button=ctk.CTkButton(app,text="Report📋",command=find_report,hover_color="green"
,fg_color="blue",border_color="white",border_width=0
                            )
report_button.place(x=350,y=780)
    
confusion_button=ctk.CTkButton(app,text="Metrix",command=find_confusion,hover_color="green"
,fg_color="blue",border_color="white",border_width=0
                               )
confusion_button.place(x=500,y=780)
    
confusion_button=ctk.CTkButton(app,text="clear🗑",command=clear,hover_color="orange"
,fg_color="blue",border_color="white",border_width=0,width=500
                               )
confusion_button.place(x=100,y=710)
  

# Results text box with frame
result_frame=ctk.CTkFrame(master=app,width=700,height=650,border_width=0)
result_frame.place(x=0,y=50)


results_textbox = ctk.CTkTextbox(result_frame, height=650, width=700, font=("Arial", 12),
border_width=0,border_color="black"
)

results_textbox.place(x=0,y=0)

#new input 
predi_frame=ctk.CTkFrame(master=app,width=800,height=750,border_width=0,border_color="black")
predi_frame.place(x=700,y=50)

predict= ctk.CTkEntry(predi_frame, placeholder_text="EnterYour Mood o_o",height=100, width=750)

predict.place(x=0,y=650)

predict_button=ctk.CTkButton(predi_frame,text="sent↗",width=50,height=100,fg_color="green",hover_color='white',text_color="black",command=get_sentiment)

predict_button.place(x=750,y=650)

predict_result=ctk.CTkTextbox(predi_frame, height=650, width=800, font=("Arial", 12),
                              
border_width=0,border_color="black")
predict_result.place(x=0,y=0)
# Run the application
app.mainloop()