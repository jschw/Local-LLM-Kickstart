import pyperclip

def get_text_clipboard():

    try:
        content = pyperclip.paste()
    except Exception:
        return None

    if content.strip() == "":
        return None
    else:
        return content


if __name__ == "__main__":
    print(get_text_clipboard())
