import threading
import asyncio
import sys
from app.utils.ws_manager import ws_manager
from app.train.train_model import _train

_main_loop: asyncio.AbstractEventLoop | None = None

def set_main_loop(loop: asyncio.AbstractEventLoop):
    global _main_loop
    _main_loop = loop

def _threaded_train(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    main_loop = _main_loop  # capture the REAL app loop

    def broadcast(msg: str):
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), main_loop)

    class StreamForwarder:
        def __init__(self, original):
            self.original = original

        def write(self, s: str):
            try:
                self.original.write(s)
                self.original.flush()
            except Exception:
                pass
            stripped = s.strip()
            if stripped:
                broadcast(stripped)
            return len(s)

        def flush(self):
            try:
                self.original.flush()
            except Exception:
                pass

    orig_stdout = sys.stdout
    sys.stdout = StreamForwarder(orig_stdout)

    try:
        broadcast(f"🔔 Training starting on thread {threading.get_ident()}")
        _train(dataset_dir=dataset_dir, epochs_to_add=epochs, imgsz=imgsz, val_ratio=val_ratio)
        broadcast("✅ Training finished")
    except Exception as e:
        broadcast(f"❌ Training failed: {str(e)}")
        raise
    finally:
        sys.stdout = orig_stdout
        broadcast("🔚 Trainer thread finished")

def start_training_thread(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    t = threading.Thread(
        target=_threaded_train,
        args=(dataset_dir, epochs, imgsz, val_ratio),
        daemon=True
    )
    t.start()
    return t