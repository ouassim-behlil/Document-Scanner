from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

class PaddleOCRProcessor:
    """
    A class to process images using PaddleOCR for text extraction and visualization.
    """

    def __init__(self, lang='en', font_path=None):
        """
        Initializes the PaddleOCRProcessor with the specified language and font path.

        Args:
            lang (str): Language for OCR (e.g., 'en' for English). Default is 'en'.
            font_path (str): Path to the font file for drawing OCR results. Default is None.
        """
        # Initialize PaddleOCR
        self.__ocr = PaddleOCR(lang=lang)  # Private variable for OCR model
        self.__font_path = font_path  # Private variable for font path

    def extract_text(self, img_path, cls=False):
        """
        Extracts text from the input image using PaddleOCR.

        Args:
            img_path (str): Path to the input image file.
            cls (bool): Whether to use text classification. Default is False.

        Returns:
            list: List of OCR results containing bounding boxes, text, and confidence scores.
        """
        # Perform OCR on the image
        result = self.__ocr.ocr(img_path, cls=cls)
        return result[0]  # Return the first batch of results

    def visualize_results(self, img_path, ocr_result, output_path):
        """
        Visualizes the OCR results by drawing bounding boxes, text, and confidence scores on the image.

        Args:
            img_path (str): Path to the input image file.
            ocr_result (list): OCR results from `extract_text`.
            output_path (str): Path to save the visualized image.
        """
        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Extract bounding boxes, text, and scores
        boxes = [line[0] for line in ocr_result]
        txts = [line[1][0] for line in ocr_result]
        scores = [line[1][1] for line in ocr_result]

        # Draw OCR results on the image
        im_show = draw_ocr(image, boxes, txts, scores, font_path=self.__font_path)

        # Convert the result to an image and save it
        im_show = Image.fromarray(im_show)
        im_show.save(output_path)
        print(f"Visualized results saved to {output_path}")

    @staticmethod
    def print_results(ocr_result):
        """
        Prints the OCR results to the console.

        Args:
            ocr_result (list): OCR results from `extract_text`.
        """
        for line in ocr_result:
            print(line)


# Example Usage
if __name__ == "__main__":
    # Paths
    img_path = "outputs/preprocessed_receipt.jpg"
    output_path = "outputs/result.jpg"
    font_path = "src/docs/french.ttf"

    try:
        # Initialize the PaddleOCRProcessor
        processor = PaddleOCRProcessor(lang='en', font_path=font_path)

        # Extract text from the image
        ocr_result = processor.extract_text(img_path)

        # Print OCR results to the console
        processor.print_results(ocr_result)

        # Visualize the OCR results and save the output
        processor.visualize_results(img_path, ocr_result, output_path)

    except Exception as e:
        print(f"Error during OCR processing: {e}")