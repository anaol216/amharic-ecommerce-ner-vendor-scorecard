from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv

# Load Telegram API credentials from .env
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# ✅ Telegram client session
client = TelegramClient('scraping_session', api_id, api_hash)

# 📦 Directory for media files
media_dir = '../Data/raw/media'
os.makedirs(media_dir, exist_ok=True)

# ✅ Full list of vendor channels to scrape
channels = [
    '@Shewabrand', '@helloomarketethiopia',
    '@modernshoppingcenter', '@qnashcom', '@Fashiontera', '@kuruwear',
    '@gebeyaadama', '@MerttEka', '@forfreemarket', '@classybrands',
    '@marakibrand', '@aradabrand2', '@marakisat2', '@belaclassic', '@AwasMart'
]

# ✨ Scraping function
async def scrape_channel(client, channel_username, writer):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        async for message in client.iter_messages(entity, limit=1000):
            if not message.message:
                continue  # skip empty messages

            media_path = ""
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username.strip('@')}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)

            writer.writerow([
                channel_title,
                channel_username,
                message.id,
                message.message.replace('\n', ' '),
                message.date,
                media_path
            ])
        print(f"Done scraping: {channel_username}")
    except Exception as e:
        print(f"Error scraping {channel_username}: {e}")

# 🚀 Main function
async def main():
    await client.start()
    os.makedirs("../Data/raw/", exist_ok=True)
    with open("../Data/raw//telegram_data.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])

        for channel in channels:
            await scrape_channel(client, channel, writer)

# Run the scraping process
if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())

