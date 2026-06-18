# Контейнер-планировщик обновления векторного индекса (Задание 6).
# Внутри образа работает системный cron, который ежедневно в 06:00 запускает
# update_index.py. Код и данные монтируются с хоста (см. docker-compose.yml),
# поэтому обновляется тот же индекс Task3/index/, который читает Telegram-бот.
FROM python:3.11-slim

# cron — системный планировщик; запускаем его в контейнере в foreground.
RUN apt-get update \
 && apt-get install -y --no-install-recommends cron \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости апдейтера (кэшируемый слой). Код приходит volume-монтированием.
COPY Task6/requirements.txt /app/Task6/requirements.txt
RUN pip install --no-cache-dir -r Task6/requirements.txt

# Модель bge-m3 кэшируется в /app/.hfcache (монтируется как именованный volume,
# чтобы не скачивать ~2 ГБ при каждом запуске).
ENV HF_HOME=/app/.hfcache \
    PYTHONUNBUFFERED=1

# Расписание в /etc/cron.d (системный формат с полем пользователя).
COPY Task6/crontab /etc/cron.d/ragupdate
RUN chmod 0644 /etc/cron.d/ragupdate

# cron в foreground — главный процесс контейнера.
CMD ["cron", "-f"]
