

# Optical-Comm-ML
Aprendizaje automático para comunicaciones ópticas a través de una fibra dispersiva

El objetivo de este proyecto es utilizar técnicas de aprendizaje automático para decodificar una señal distorsionada recibida a través de una fibra óptica simulada.\
Examinamos dos codificaciones: binaria (NRZ) y de 4 bits (4-PAM)

Estimador de Máxima Verosimilitud (Matlab)\
Asumimos que cada muestra de tiempo dentro de un ciclo de reloj tiene una distribución gaussiana.\
Una muestra de entrada se escala por la varianza para encontrar la desviación respecto a la media de cada símbolo, y la predicción se determina mediante el símbolo más cercano.

Red Neuronal Convolucional (Python)\
Una CNN típicamente utilizada para la clasificación de imágenes 2D se modifica para clasificar formas de onda 1D.\
En lugar de convoluciones 2D, utilizamos convoluciones causales dilatadas 1D.

Máquina de Vectores de Soporte (Matlab)\
La señal en el dominio del tiempo se divide en conjuntos de muestras basados en la tasa de bits.\
El SVM se entrena con las muestras después de que hayan sido convertidas en vectores de características de 16 dimensiones, donde cada dimensión es una muestra dentro del ciclo de reloj.

Dependencias
>Matlab: Ninguna\
>Python: Python 3.6, Numpy 1.14, TensorFlow 1.6

Archivos de extracción de características\
auto_labeler.m - ejecutar el programa para generar un archivo de etiquetas a partir del archivo TX\
data_parser.m - función para analizar los archivos RX y de etiquetas en matrices de entrenamiento y prueba

Archivos del Estimador de Máxima Verosimilitud\
MLbin.m - ejecuta la máxima verosimilitud sobre datos binarios\
MLpam.m - ejecuta la máxima verosimilitud sobre datos 4-PAM

Archivos de la Red Neuronal Convolucional\
dilated_neural_net.py - define funciones auxiliares para entrenar la CNN\
test_nn_binary.py - ejecutar para entrenar la CNN a fin de clasificar datos binarios\
test_nn_pam4.py - ejecutar para entrenar la CNN a fin de clasificar datos 4-PAM\
Para ejecutar las visualizaciones, utilice el siguiente comando (alternativamente python -m tensorboard.main)
>tensorboard --logdir=/tmp/dilated_cnn_model_binary

o bien

>tensorboard --logdir=/tmp/dilated_cnn_model_pam4

Una vez que TensorBoard esté en ejecución, navegue con su navegador web a localhost:6006 para ver el TensorBoard.


Archivos de la Máquina de Vectores de Soporte\
SVM_train.m - función para entrenar un hiperplano SVM utilizando la matriz de entrenamiento e hiperparámetros\
SVM_test.m - función para evaluar un hiperplano SVM utilizando la matriz de prueba e hiperparámetros\
binary_SVM.m - ejecutar SVM para datos RX binarios\
svmp4.m - ejecutar SVM para datos RX 4-PAM

svmp4_helper.m - Llama a svmp4.m N veces y devuelve promedios\
binary_SVM_helper.m - Llama a binary_SVM.m N veces y devuelve promedios\
hard_margin_main.m - Recopila y grafica datos para SVM binario y 4-PAM\
test_regularizer.m - ejecutar el programa para evaluar la SVM de margen blando\
test_desynch.m - ejecutar el programa para evaluar los efectos de la desincronización

Archivos del Núcleo de Base Radial Gaussiana\
kernelbin_SVM.m - esta función contiene el código del núcleo RBF para el entrenamiento y evaluación de datos binarios\
kernelpam_SVM.m - esta función contiene el código del núcleo RBF para el entrenamiento y evaluación de datos 4PAM\
kernelfindparam.m - archivo .m que implementa el núcleo binario. La variable Oneshot determina si se debe realizar un barrido de ajuste utilizando vectores variables; de lo contrario, la función se ejecutará una vez con nuestros valores predeterminados ajustados.\
kernelfindparampam.m  - archivo .m que implementa el núcleo 4PAM.
