# Лабораторная работа #2
## Обучить нейронную сеть EfficientNet-B0 (случайное начальное приближение) для решения задачи классификации изображений Food-101 [6]
### Изначальные результаты обучения
Оранжевая кривая отображает обучающую выборку, синяя - валидационную.
https://tensorboard.dev/experiment/PsRtllF9Q2Kb1hXNi0eZDQ/#scalars
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab2/main/epoch_categorical_accuracy_1.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab2/main/epoch_loss_1.svg">

**Описание архитектуры:**
 
* Размерность входного изображения: 
```
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
```

* Использование сети EfficientNetB0
 ```
outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
 ```
 ```
* include_top=True - используем верхний слой сети EfficientNetB0
* weights=None - случайное приближение в весах
* classes = NUM_CLASSES - 101 класс в классификаторе
```

### Анализ результатов:

На графиках наблюдается практически полное отстутсвие обучаемости сети. На 3-4 эпохах наблюдается самая высокая точность (~30%), после чего она только убывает, что свидетельствует о том, что сеть не работает в случае со случайным приближением.

## С использованием и техники обучения Transfer Learning обучить нейроннуюсеть EfficientNet-B0 (предобученную на базе изображений imagenet) для решениязадачи классификации изображений Food-101

https://tensorboard.dev/experiment/RBbdEdc9RimxIzkZrk0Jqw/#scalars   
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab2/main/epoch_categorical_accuracy_2.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab2/main/epoch_loss_2.svg">

**Описание архитектуры:**
 
* Размерность входного изображения: 
```
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
```

* Использование сети EfficientNetB0
 ```
x = EfficientNetB0(include_top=False, weights='imagenet', input_tensor = inputs)
 ```
 ```
* include_top=False - не используем верхний слой сети EfficientNetB0
* weights='imagenet' - используем веса из предобученной сети EfficientNetB0 на датасете imagenet
* input_tensor=inputs - используем тензор inputs
* x.trainable = False - отключаем параметр trainable, потому что нам необходимо не учить сеть с нуля, а лишь обучить классификатор
* x = tf.keras.layers.GlobalAveragePooling2D()(x.output) - добавление слоя GlobalAveragePooling2D для уменьшения формы входного тензора
```
## Анализ результатов
Во втором случае мы можем наблюдать стабильное обучение сети примерно до 7-8 эпохи, после чего график функции потерь начинает расти, а сеть, в свою очередь, переобучаться. В общем можно сделать вывод, что использование такой технологии, как Transfer Learning позволяет увеличить точность (до ~67% по сравнению с предыдущим результатом в ~30%). В первом же случае можно с большой уверенностью утверждать, что сеть практически не обучается.
