from app.database import DatabaseManager
import os

try:
    db = DatabaseManager()
    db.initialize_tables()
    print("✅ ¡Conexión exitosa! El archivo .db debería aparecer en tu Drive.")
    print(f"Ruta usada: {os.getenv('DB_PATH')}")
except Exception as e:
    print(f"❌ Error en la prueba: {e}")