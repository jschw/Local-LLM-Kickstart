import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse


def crawl_website(url: str, timeout: int, max_depth: int = 1):
    """
    Recursively crawl a website starting from `url` up to `max_depth` link depth.
    Returns a list of (markdown_content, url) tuples for all visited pages.
    """

    DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
    strip_elements = ['a']
    visited = set()
    results = []

    def _crawl(current_url, depth):
        if depth > max_depth or current_url in visited:
            return
        visited.add(current_url)
        try:
            print(f"--> Crawling: {current_url} (depth {depth})")
            response = requests.get(current_url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"-->  Request error for {current_url}: {e}")
            return

        content_type = response.headers.get('Content-Type', '')

        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(['script', 'style']):
                script.decompose()

            max_text_length = 0
            main_content = ""
            for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
                text_length = len(tag.get_text())
                if text_length > max_text_length:
                    max_text_length = text_length
                    main_content = tag

            content = str(main_content)
            if len(content) == 0:
                return

            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            output = output.replace('\n','')
            output = output.replace('\t','')
            output = output.strip()
            if output:
                results.append((output, current_url))

            # Recursively crawl links if depth < max_depth
            if depth < max_depth:
                links = set()
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    # Only follow http(s) links, resolve relative URLs
                    joined = urljoin(current_url, href)
                    parsed = urlparse(joined)
                    if parsed.scheme in ['http', 'https']:
                        links.add(joined)

                for link in links:
                    print(f"Parsing sublink: {link}")
                    _crawl(link, depth + 1)

        elif 'text/plain' in content_type:
            if len(response.text) == 0:
                return
            output = md(
                response.text,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )
            output = output.replace('\n','')
            output = output.replace('\t','')
            output = output.strip()
            if output:
                results.append((output, current_url))

        elif 'application/pdf' in content_type:
            # TODO: PDF crawling not implemented
            return
        else:
            print(f"--> Unknown content type for {current_url}.")
            return

    _crawl(url, 1)
    return results
