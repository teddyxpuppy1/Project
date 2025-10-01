import threading
import webview
from kivy.app import App
from kivy.uix.button import Button

# Start Flask server in a separate thread
def start_flask():
    import os
    os.system("C:\Project\Project/app.py")  # Replace with your Flask app path

class WebApp(App):
    def build(self):
        # Button to open local Flask app in WebView
        btn = Button(text="Open Local Flask App")
        btn.bind(on_press=self.open_web)
        return btn

    def open_web(self, instance):
        # Opens the local Flask app in WebView
        webview.create_window("My Flask App", "http://127.0.0.1:5000")
        webview.start()

if __name__ == "__main__":
    # Start Flask app in background
    threading.Thread(target=start_flask).start()
    WebApp().run()
