# miniGPT
Здесь будет написана мини llm на 30 млн параметров   
Архитектура основана на чем-то среднем между gpt 2 и LLaMa 2   
Обучаться будет на openwebtext на rtx 3060, 32 gb ram, intel i5(11 поколения)    
Архитектурные решения: ротационные поизиционные эмбеддинги, kv-кэширование, MLP с SwiGLU, DropPath, echo, stop_token.
