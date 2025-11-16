import threading
import sys
import io
import asyncio
from app.ws_manager import ws_manager
from app.train_model import _train

class StreamForwarder(io.TextIOBase):
    def __init__(self, original):
        self.original = original

    def write(self, s: str) -> int:
        length = len(s)
        try:
            self.original.write(s)
            self.original.flush()
        except Exception:
            pass

        for line in s.splitlines():
            stripped = line.rstrip("\n")
            if stripped:
                try:
                    loop = asyncio.get_event_loop()
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(stripped), loop)
                    else:
                        asyncio.run(ws_manager.broadcast(stripped))
                except RuntimeError:
                    try:
                        asyncio.run(ws_manager.broadcast(stripped))
                    except Exception:
                        pass
        return length

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass


def _threaded_train(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    orig_stdout = sys.stdout
    sys.stdout = StreamForwarder(orig_stdout)
    try:
        asyncio.run(ws_manager.broadcast(f"🔔 Trainer starting at thread {threading.get_ident()}"))
        try:
            _train(dataset_dir, epochs, imgsz, val_ratio)
            asyncio.run(ws_manager.broadcast(f"✅ Training finished"))
        except Exception as e:
            asyncio.run(ws_manager.broadcast(f"❌ Training failed: {str(e)}"))
            raise
    finally:
        sys.stdout = orig_stdout
        try:
            asyncio.run(ws_manager.broadcast("🔚 Trainer thread finished"))
        except Exception:
            pass


def start_training_thread(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    t = threading.Thread(
        target=_threaded_train,
        args=(dataset_dir, epochs, imgsz, val_ratio),
        daemon=True
    )
    t.start()
    return t
