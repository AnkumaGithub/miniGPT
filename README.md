# miniGPT
Здесь написана gpt модель примерно на 150 млн параметров и дообучена gpt2 на том же датасете    
Архитектура близка к llama 2      
Обучалась на rtx 3060(12 gb)    
Датасет Tinysories - цель генерация небольших историй(до 300 токенов).    
Датасет - https://huggingface.co/datasets/roneneldan/TinyStories    
Для взаимодействия с моделями разработано c++ приложение - https://github.com/AnkumaGithub/miniGPT_app
