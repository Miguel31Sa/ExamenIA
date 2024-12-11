import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_model():
    # Directorios de las imágenes 
    train_dir = r'C:\proyectos\clasificacion_caries\dataset\train'
    validation_dir = r'C:\proyectos\clasificacion_caries\dataset\validation'

    # Preprocesamiento de las imágenes 
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Generadores de datos para entrenamiento y validación
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'  
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  
    ])

    # Compilación del modelo
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(learning_rate=0.0001),  
        metrics=['accuracy']
    )

    # Entrenar el modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=20,  
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

  
    model.save('modelo_caridad.h5')
    print("Modelo entrenado y guardado como 'modelo_caridad.h5'")

if __name__ == '__main__':
    train_model()
