import telebot
import tensorflow as tf
from PIL import Image
import io
import requests
import numpy as np

BOT_TOKEN = '7882975454:AAH0LWioYcybDGNQj80aUecxOeEL0vdcz_k'
bot = telebot.TeleBot(BOT_TOKEN)

# Загрузка модели.  ВНИМАНИЕ:  Нужно заменить на ВАШУ модель, обученную на 3 класса
model = tf.keras.models.load_model('model.h5')  # Замените!
class_names = ['человек', 'животное', 'предмет']  # Замените, если порядок другой

# Функция для предварительной обработки изображения
def preprocess_image(image):
    image = image.resize((128, 128))  # Измените размер под вашу модель
    image = np.array(image)
    image = image / 255.0  # Нормализация (важно для многих моделей)
    image = np.expand_dims(image, axis=0) # Добавляем размерность пакета
    return image

# Функция для предсказания класса изображения
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return class_names[predicted_class], confidence


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправьте мне изображение, и я попробую определить, что на нем изображено (человек, животное или предмет).")

# Обработчик изображений
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded_file))

        predicted_class, confidence = predict_image(image)

        bot.reply_to(message, f"Я думаю, что это: {predicted_class} (уверенность: {confidence:.2f})")

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")

# Обработчик текста (URL)
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    try:
        response = requests.get(message.text, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))

        predicted_class, confidence = predict_image(image)

        bot.reply_to(message, f"Я думаю, что это: {predicted_class} (уверенность: {confidence:.2f})")

    except requests.exceptions.RequestException as e:
        bot.reply_to(message, "Не удалось скачать изображение по указанному URL.")

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка при обработке URL: {e}")

# Запуск бота
if __name__ == '__main__':
    bot.infinity_polling()
