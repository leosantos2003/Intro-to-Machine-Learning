# Intro to Machine Learning
  
## About

This repository documents the learnings from completing the [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course on [Kaggle](https://www.kaggle.com/).

Hands-on project focused on establishing a solid foundation in the workflow of a Machine Learning project, from data exploration to model validation.

* `lesson_2`: basic data exploration; loading and understanding data.

* `lesson_3`: first machine learning model; building the first model.
 
* `lesson_4`: model validation; measuring the performance of the model for testing and comparing alternatives. [(see graphics)](https://github.com/leosantos2003/Intro-to-Machine-Learning/tree/main/lesson_4)
 
* `lesson_5`: underfitting and overfitting; fine-tuning the model for better performance. [(see graphics)](https://github.com/leosantos2003/Intro-to-Machine-Learning/tree/main/lesson_5)
 
* `lesson_6`: random forests; using a more sophisticated machine learning algorithm. [(see graphic)](https://github.com/leosantos2003/Intro-to-Machine-Learning/tree/main/lesson_6)

<div align="center">
  
  ## Lesson 4

</div>

`lesson 4`: model validation; measuring the performance of the model for testing and comparing alternatives.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* Comparison between Prediction Values and Actual Values.
* Model trained with full DataSet and put to predict already known values from its own training. That explains why the values match so perfectly.

</div>
      <img width="600" height="360" alt="comparison_in_sample" src="https://github.com/user-attachments/assets/f2df2f7f-c1d8-46fd-a93e-42f3857794e7" />
<div class="texto-titulo">
            
### Graphic 2:
* Comparison between Prediction Values and Actual Values.
* Model trained with half of the DataSet and put to predict the other half. The strategy of splitting up the DataSet makes the predictions more reliable.
            
</div>
      <img width="600" height="360" alt="comparison_validation" src="https://github.com/user-attachments/assets/4d125f16-ce47-40dd-9092-21abf88b4cb2" />
</div>

<div align="center">
  
  ## Lesson 5

</div>

`lesson_5`: underfitting and overfitting; fine-tuning the model for better performance.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* Shows the relations between Model Performance and its respective Decision Tree Size.
* It's important to determine the tree size for the lowest Mean Absolute Error. A tree too big or too low may not be so precise in most cases. 

</div>
      <img width="600" height="360" alt="mae_vs_leaf_nodes" src="https://github.com/user-attachments/assets/bd408789-c4c8-4938-9bc1-15068a811dea" />
<div class="texto-titulo">
      
### Graphic 2:
* Comparison between Prediction Values and Actual Values when using the best tree size.
* Choosing the optimal tree size caused the predictions to be very precise, given that the Mean Absolute Error was the lowest.

</div>
      <img width="600" height="360" alt="final_model_comparison" src="https://github.com/user-attachments/assets/fc185c9f-569c-4fbe-936f-8fd38687d637" />
</div>

<div align="center">
  
  ## Lesson 6

</div>

`lesson_6`: random forests; using a more sophisticated machine learning algorithm.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* Compares the Mean Absolute Error between three models: a simple model with generic Decision Tree, a model with Decision Tree with optimized tree size, and a model with Random Forest.
* The comparision shows the superiority of Random Forest over the Decision Tree in any case. 

</div>
      <img width="600" height="360" alt="mae_models_comparison" src="https://github.com/user-attachments/assets/19926981-f8c1-4ae5-8671-d630b43b148a" />
</div>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
