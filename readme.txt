"# sourcecode"

Compute_second_grad.py : compute the second gradient of the model after training
MobileNetV2.py : the model structure of MobileNetV2
MobileNetV2_main.py : train method and test method of MobileNetV2, after train save the model
MyModel_cifar.py : Model that trained on CIFAR dataset
MyModel_MNIST.py : Model that trained on MNIST dataset
MyModel_train_model.py : train MyModel_cifar and MyModel_MNIST then save the model
test_kmeans_quantiz : use different level to quantiz the model and save the result of before and after quantiz loss and accurancy
use_degree_kmeans.py : save the most importent parameters and quantiz the remaining parameters

folders:
Kmeans_quantiz_use_hassian : the results of use_degree_kmeans.py
old_source_code : some old source codes and test codes, doesn't matter
result_of_quantiz_after_training_without_curvature : the result of test_kmeans_quantiz.py
saved_models_after_training : the saved models that training without curvature information
