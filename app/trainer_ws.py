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
    import sys, threading
    from app.train_model import _train
    import asyncio
    from app.ws_manager import ws_manager

    class StreamForwarder:
        def __init__(self, original):
            self.original = original

        def write(self, s: str):
            try:
                self.original.write(s)
                self.original.flush()
            except Exception:
                pass
            # forward to WS
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast(s.strip()), loop)
            except Exception:
                pass
            return len(s)

        def flush(self):
            try:
                self.original.flush()
            except Exception:
                pass

    # Redirect stdout
    orig_stdout = sys.stdout
    sys.stdout = StreamForwarder(orig_stdout)

    loop = None  # <-- define before try
    try:
        # Each thread gets its own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Broadcast start
        loop.run_until_complete(ws_manager.broadcast(f"🔔 Trainer starting at thread {threading.get_ident()}"))

        # Run training
        try:
            _train(dataset_dir=dataset_dir, epochs=epochs, imgsz=imgsz, val_ratio=val_ratio)
            loop.run_until_complete(ws_manager.broadcast("✅ Training finished"))
        except Exception as e:
            loop.run_until_complete(ws_manager.broadcast(f"❌ Training failed: {str(e)}"))
            raise

    finally:
        # Restore stdout
        sys.stdout = orig_stdout
        if loop:
            try:
                loop.run_until_complete(ws_manager.broadcast("🔚 Trainer thread finished"))
            finally:
                loop.close()


def start_training_thread(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    t = threading.Thread(
        target=_threaded_train,
        args=(dataset_dir, epochs, imgsz, val_ratio),
        daemon=True
    )
    t.start()
    return t
