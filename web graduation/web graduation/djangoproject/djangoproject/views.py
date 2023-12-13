#def my_view(request):
 #   if request.method == 'POST':
  #      my_file = request.FILES['my_file']
        # Read the contents of the file
   #     file_contents = my_file.read()
         # Do something with the file contents

from django.shortcuts import render
# from django.http import response,request
#from ml.predict import predict_label

def home(request):
     return render(request,'index.html')

def Diagnosis(request):
    string_value = ''  # Initialize string_value with an empty string

    if request.method == 'POST':
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression

        try:
            # Read the uploaded file and convert to NumPy array
            uploaded_file = request.FILES['file']
            file_content = uploaded_file.read().decode('utf-8')
            test_input = np.array(file_content.split())

            with open('ML_classes.txt', 'r') as f:
                y = np.array([line.strip() for line in f], dtype=object)
            X = np.loadtxt("Features.txt")

            # Create a StratifiedKFold object with 5 splits
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train and evaluate Logistic Regression
                lr_model = LogisticRegression(max_iter=1000)
                lr_model.fit(X_train, y_train)

            # Reshape the input data and make a prediction
            X_sample = test_input.reshape(1, -1)
            y_pred = lr_model.predict(X_sample)
            string_value = y_pred[0]

            print(file_content)

        except Exception as e:
            # Handle any errors that occur during the execution of the code
            print("An error occurred: ", e)

    # Use the value of string_value
    print("The predicted value is: ", string_value)
    return render(request,'Diagnosis.html',{'file_content': string_value})

def Research(request):
     return render(request,'Research.html')

def services(request):
     return render(request,'services.html')

def care(request):
     return render(request,'care.html')




#def upload_file(request):
 #   if request.method == 'POST':
  #      uploaded_file = request.FILES['csv_file']
   #     with open('uploaded_file.csv', 'wb+') as destination:
    #        for chunk in uploaded_file.chunks():
     #           destination.write(chunk)
      #  return render(request, 'upload_success.html')
   # return render(request, 'upload.html')

