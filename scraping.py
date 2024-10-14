from playwright.sync_api import sync_playwright
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup


def get_medication_info(medication_name):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set headless=True to run in the background
        page = browser.new_page()

        # Step 1: Go to the website
        page.goto("https://base-donnees-publique.medicaments.gouv.fr/")

        # Step 2: Fill out the search field
        search_input = page.locator("input#txtCaracteres")
        search_input.fill(medication_name)

        # Step 3: Submit the form (simulate pressing enter)
        search_input.press("Enter")

        # Step 4: Wait for the results to load and click on the first result
        page.wait_for_selector("table.result")  # Adjust the selector if needed

        # page.pause()

        # Step 5: Extract the first medication link
        first_medicine = page.locator("table.result tbody tr").nth(1)  # Use nth(1) to skip the header row
        
        # Locate the nested <a> element
        link_element = first_medicine.locator("td.ResultRowDeno a.standart")

        if link_element.count() > 0:
            name = link_element.inner_text().strip()
            link = link_element.get_attribute("href")
            print(f"Name: {name}, Link: {link}")
        else:
            print("No results found.")

        url = f"https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?{link.split('?')[1]}&typedoc=R"

        loader = WebBaseLoader(url)
        docs = loader.load()

        # Get the HTML content
        html_content = docs[0].page_content  # Adjust based on how you want to access the content

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract raw text from the HTML
        raw_text = soup.get_text(separator=' ', strip=True)

        # Print the raw text
        return {
            "medication_info": raw_text,
            "link": url
        }
