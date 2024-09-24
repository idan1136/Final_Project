import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# Paths to the dataset
base_dir = '45-76_gaf_images'
accident_dir = os.path.join(base_dir, 'ACCIDENT')
not_accident_dir = os.path.join(base_dir, 'NOT ACCIDENT')


# Data generators
def create_data_generators(accident_dir, not_accident_dir, img_size=(200, 200), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.3,
                                       horizontal_flip=True, fill_mode='nearest', validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    def create_generators_for_axis(axis):
        train_generator = train_datagen.flow_from_directory(
            base_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training')
        val_generator = train_datagen.flow_from_directory(
            base_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation')

        test_generator = test_datagen.flow_from_directory(
            base_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

        return train_generator, val_generator, test_generator

    return create_generators_for_axis('X'), create_generators_for_axis('Y'), create_generators_for_axis('Z')

# Create data generators for each axis
(axis_X_generators, axis_Y_generators, axis_Z_generators) = create_data_generators(accident_dir, not_accident_dir)

(train_gen_X, val_gen_X, test_gen_X) = axis_X_generators
(train_gen_Y, val_gen_Y, test_gen_Y) = axis_Y_generators
(train_gen_Z, val_gen_Z, test_gen_Z) = axis_Z_generators

# CNN model architecture
def build_model(input_shape=(200, 200, 3)):
    model = Sequential()

    # First Conv Block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout to prevent overfitting

    # Second Conv Block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout to prevent overfitting

    # Third Conv Block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout to prevent overfitting

    # Fourth Conv Block
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout to prevent overfitting

    # Flatten the layers to pass to a dense layer
    model.add(Flatten())

    # Fully connected Dense layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  # Increased Dropout here

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model
# Correct class weights based on provided data
total_accident = 10347
total_not_accident = 24213

class_weight = {
    0: total_accident / (total_accident + total_not_accident),
    1: total_not_accident / (total_accident + total_not_accident)
}

# Train and save model with early stopping
def train_and_save_model(train_gen, val_gen, model_save_path, axis):
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.002), loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=[checkpoint, reduce_lr, early_stopping],
        class_weight=class_weight
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(base_dir, f'training_history_{axis}.csv'), index=False)

    # Save training accuracy and loss plots
    plot_training_history(history, axis)

    return model, history

# Plot training history for each axis
def plot_training_history(history, axis):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - {axis}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save accuracy plot
    plt.savefig(os.path.join(base_dir, f'training_accuracy_{axis}.png'))

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {axis}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save loss plot
    plt.savefig(os.path.join(base_dir, f'training_loss_{axis}.png'))

    plt.close()

# Train and save models for each axis
model_X, history_X = train_and_save_model(train_gen_X, val_gen_X, os.path.join(base_dir, 'model_X.keras'), 'X')
model_Y, history_Y = train_and_save_model(train_gen_Y, val_gen_Y, os.path.join(base_dir, 'model_Y.keras'), 'Y')
model_Z, history_Z = train_and_save_model(train_gen_Z, val_gen_Z, os.path.join(base_dir, 'model_Z.keras'), 'Z')

# Evaluate model and save confusion matrix as an image
def evaluate_model(test_gen, model_path, axis):
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    y_pred = (model.predict(test_gen) > 0.5).astype("int32")
    y_true = test_gen.classes

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Non-Accident', 'Accident'],
                         columns=['Predicted Non-Accident', 'Predicted Accident'])
    cm_df.to_csv(os.path.join(base_dir, f'confusion_matrix_{axis}.csv'))

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {axis}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(base_dir, f'confusion_matrix_{axis}.png'))
    plt.close()

    # Classification report
    cr = classification_report(y_true, y_pred, target_names=['Non-Accident', 'Accident'], output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(base_dir, f'classification_report_{axis}.csv'))

    # F1 score
    f1 = f1_score(y_true, y_pred)

    return test_acc, f1, cm

# Evaluate models for each axis
results_X = evaluate_model(test_gen_X, os.path.join(base_dir, 'model_X.keras'), 'X')
results_Y = evaluate_model(test_gen_Y, os.path.join(base_dir, 'model_Y.keras'), 'Y')
results_Z = evaluate_model(test_gen_Z, os.path.join(base_dir, 'model_Z.keras'), 'Z')

# Save evaluation results
results_df = pd.DataFrame({
    'Axis': ['X', 'Y', 'Z'],
    'Test Accuracy': [results_X[0], results_Y[0], results_Z[0]],
    'F1 Score': [results_X[1], results_Y[1], results_Z[1]]
})
results_df.to_csv(os.path.join(base_dir, 'evaluation_results.csv'), index=False)

# Majority voting implementation
def majority_voting(test_gens, model_paths):
    preds = []
    for model_path, test_gen in zip(model_paths, test_gens):
        model = load_model(model_path)
        preds.append((model.predict(test_gen) > 0.5).astype("int32"))

    preds = np.array(preds)
    majority_vote = np.sum(preds, axis=0) >= 2  # Majority voting threshold

    y_true = np.concatenate([gen.classes for gen in test_gens])
    return majority_vote, y_true

# Perform majority voting
test_generators = [test_gen_X, test_gen_Y, test_gen_Z]
model_paths = [os.path.join(base_dir, 'model_X.keras'), os.path.join(base_dir, 'model_Y.keras'),
               os.path.join(base_dir, 'model_Z.keras')]
y_pred_majority, y_true = majority_voting(test_generators, model_paths)

# Evaluate majority voting
cm_majority = confusion_matrix(y_true, y_pred_majority)
cr_majority = classification_report(y_true, y_pred_majority, target_names=['Non-Accident', 'Accident'],
                                    output_dict=True)
f1_majority = f1_score(y_true, y_pred_majority)

# Save majority voting results
cm_majority_df = pd.DataFrame(cm_majority, index=['Non-Accident', 'Accident'],
                              columns=['Predicted Non-Accident', 'Predicted Accident'])
cm_majority_df.to_csv(os.path.join(base_dir, 'confusion_matrix_majority.csv'))

# Plot and save confusion matrix for majority voting
plt.figure(figsize=(8, 6))
sns.heatmap(cm_majority_df, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - Majority Voting')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(base_dir, 'confusion_matrix_majority.png'))
plt.close()

cr_majority_df = pd.DataFrame(cr_majority).transpose()
cr_majority_df.to_csv(os.path.join(base_dir, 'classification_report_majority.csv'))

results_majority_df = pd.DataFrame({
    'Method': ['Majority Voting'],
    'Test Accuracy': [np.mean(y_true == y_pred_majority)],
    'F1 Score': [f1_majority]
})
results_majority_df.to_csv(os.path.join(base_dir, 'evaluation_results_majority.csv'), index=False)

# Plot ROC curves for each axis
def plot_roc_curve(y_true, y_pred, axis, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {axis}')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# Plot ROC curves for each axis
for axis, test_generator, model_path in zip(['X', 'Y', 'Z'], [test_gen_X, test_gen_Y, test_gen_Z], model_paths):
    model = load_model(model_path)
    y_pred_prob = model.predict(test_generator)
    plot_roc_curve(test_generator.classes, y_pred_prob, axis, os.path.join(base_dir, f'roc_curve_{axis}.png'))

# Plot ROC curve for majority voting
y_pred_prob_majority = np.mean([load_model(model_path).predict(test_generator) for model_path, test_generator in
                                zip(model_paths, [test_gen_X, test_gen_Y, test_gen_Z])], axis=0)
plot_roc_curve(y_true, y_pred_prob_majority, 'Majority', os.path.join(base_dir, 'roc_curve_majority.png'))
