# app/trainer_ws.py
import threading
import sys
import io
import asyncio
from app.ws_manager import ws_manager
from app.train_model import train_yolo_autosplit  # adjust import path if needed

class StreamForwarder(io.TextIOBase):
    """
    Lightweight stream that writes to both original stdout and the ws_manager buffer.
    """
    def __init__(self, original):
        self.original = original

    def write(self, s: str) -> int:
        # Always return the number of characters written (required by TextIOBase)
        length = len(s)

        # Write to original stdout
        try:
            self.original.write(s)
            self.original.flush()
        except Exception:
            pass

        # Forward to websocket
        lines = s.splitlines()
        for line in lines:
            stripped = line.rstrip("\n")
            if stripped:
                try:
                    loop = asyncio.get_event_loop()
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            ws_manager.broadcast(str(stripped)), loop
                        )
                    else:
                        asyncio.run(ws_manager.broadcast(str(stripped)))
                except RuntimeError:
                    try:
                        asyncio.run(ws_manager.broadcast(str(stripped)))
                    except Exception:
                        pass

        return length  # ✅ required to satisfy io.TextIOBase.write()

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

def _threaded_train(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    """
    Runs train_yolo_autosplit while redirecting stdout -> ws_manager.
    This function is intended to run in a separate daemon thread.
    """
    orig_stdout = sys.stdout
    forwarder = StreamForwarder(orig_stdout)
    sys.stdout = forwarder
    try:
        # Inform clients
        asyncio.run(ws_manager.broadcast(f"🔔 Trainer starting at thread {threading.get_ident()}"))

        # run the train function from your existing train_model.py
        try:
            best = train_yolo_autosplit(
                dataset_dir=dataset_dir,
                epochs=epochs,
                imgsz=imgsz,
                val_ratio=val_ratio
            )
            asyncio.run(ws_manager.broadcast(f"✅ Training finished. Best weights: {best}"))
        except Exception as e:
            # ensure exception messages are forwarded
            asyncio.run(ws_manager.broadcast(f"❌ Training failed: {str(e)}"))
            raise
    finally:
        # restore stdout
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
