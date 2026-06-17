import os
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from fpdf import FPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask App Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained X-ray classification model
try:
    model = tf.keras.models.load_model('x-ray-classification.h5')
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Configure Gemini API with API Key from environment
try:
    api_key = os.environ.get('API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured successfully!")
    else:
        print("⚠️  Warning: API_KEY not found in .env file. PDF reports will use default template.")
except Exception as e:
    print(f"⚠️  Warning: Could not configure Gemini API: {e}. PDF reports will use default template.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- Grad-CAM CLASS --------------------
class GradCAM:
    def __init__(self, model, class_idx=0):
        self.model = model
        self.class_idx = class_idx
        self.last_conv_layer = self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        # First, try to find conv layers in the main model
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        
        # If not found, look inside nested models (for transfer learning models)
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):  # Check if it's a nested model
                for nested_layer in reversed(layer.layers):
                    if 'conv' in nested_layer.name.lower():
                        return nested_layer.name
        
        # Last resort: use VGG16's last conv layer name
        return 'block5_conv3'

    def compute_heatmap(self, img_array):
        # For transfer learning models, we need to get the base model
        try:
            # Try to access the base model (VGG16) if it exists
            base_model = None
            for layer in self.model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    base_model = layer
                    break
            
            if base_model:
                # Use the base model for grad-cam
                grad_model = tf.keras.models.Model(
                    [self.model.inputs],
                    [base_model.get_layer(self.last_conv_layer).output, self.model.output]
                )
            else:
                # Use the regular model
                grad_model = tf.keras.models.Model(
                    [self.model.inputs],
                    [self.model.get_layer(self.last_conv_layer).output, self.model.output]
                )

            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_array)
                predictions = tf.convert_to_tensor(predictions)

                if len(predictions.shape) > 1:
                    loss = predictions[:, self.class_idx]
                else:
                    loss = predictions

            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            return heatmap.numpy()
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            # Return a dummy heatmap if grad-cam fails
            return np.zeros((7, 7))

# -------------------- REPORT CREATION --------------------
def create_pdf_report(prediction, grad_cam_path, detailed_report):
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("DejaVu", size=12)

    pdf.cell(200, 10, txt="Pneumonia Diagnosis Report", ln=True, align='C')

    prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction
    pdf.cell(200, 10, txt=f"Diagnosis: {'Pneumonia Detected' if prediction_value > 0.5 else 'No Pneumonia'}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Confidence: {prediction_value:.2f}", ln=True)

    if grad_cam_path and os.path.exists(grad_cam_path):
        pdf.cell(200, 10, txt="Grad-CAM Visualization:", ln=True)
        pdf.image(grad_cam_path, x=10, y=None, w=190)

    pdf.cell(200, 10, txt="Detailed Diagnosis Report:", ln=True)

    detailed_report = detailed_report.encode('ascii', 'replace').decode('ascii')
    pdf.multi_cell(0, 10, detailed_report)

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "diagnosis_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

# -------------------- GEMINI REPORT GENERATION --------------------
def generate_detailed_report(prediction, additional_info):
    try:
        prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction
        
        # Use the correct model name for Gemini API
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        prompt = f"""
        A patient has undergone a chest X-ray analysis using an AI-powered diagnostic system.
        
        Diagnosis Result: {'Pneumonia Detected' if prediction_value > 0.5 else 'No Pneumonia Detected'}.
        Prediction Confidence: {prediction_value:.2%}.
        Additional Analysis Information: {additional_info}.
        
        Please generate a detailed medical report that includes:
        1. Summary of Findings
        2. Clinical Interpretation
        3. Key Observations from the X-ray analysis
        4. Recommended Next Steps
        5. Important Disclaimers (AI-assisted diagnosis should be confirmed by medical professionals)
        
        Keep the report professional, clear, and concise (maximum 300 words).
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        # Return a default report if API fails
        prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction
        diagnosis = 'Pneumonia Detected' if prediction_value > 0.5 else 'No Pneumonia Detected'
        
        default_report = f"""
CHEST X-RAY ANALYSIS REPORT

FINDINGS:
The AI-powered diagnostic system has analyzed the chest X-ray image and produced the following results:

Diagnosis: {diagnosis}
Confidence Level: {prediction_value:.1%}

CLINICAL INTERPRETATION:
{'The analysis indicates the presence of pneumonia-like patterns in the chest X-ray. Areas of consolidation or infiltrates have been detected that are consistent with pneumonic changes.' if prediction_value > 0.5 else 'The chest X-ray appears normal with no significant abnormalities detected. The lung fields are clear without evidence of consolidation or infiltrates.'}

RECOMMENDATIONS:
{'- Immediate clinical correlation is recommended\n- Consider additional diagnostic tests if symptoms persist\n- Follow-up imaging may be necessary\n- Antibiotic therapy may be indicated based on clinical presentation' if prediction_value > 0.5 else '- Continue routine monitoring if asymptomatic\n- Follow standard preventive care guidelines\n- Seek medical attention if respiratory symptoms develop\n- Schedule follow-up as per standard protocols'}

IMPORTANT DISCLAIMER:
This report is generated by an AI-assisted diagnostic tool and should NOT be used as the sole basis for medical decisions. All findings must be reviewed and confirmed by a qualified radiologist or healthcare professional. This system is intended to assist, not replace, professional medical judgment.

Report Generated: {np.datetime64('now')}
Analysis Method: Deep Learning CNN with Transfer Learning (VGG16)
        """
        return default_report

# -------------------- GRAD-CAM GENERATION --------------------
def generate_grad_cam(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    grad_cam = GradCAM(model, class_idx=0)
    heatmap = grad_cam.compute_heatmap(img_array)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize(tf.expand_dims(heatmap, -1), (img.shape[0], img.shape[1])).numpy()
    heatmap = np.squeeze(heatmap)
    colormap = plt.cm.jet(heatmap)[:, :, :3]
    colormap = np.uint8(255 * colormap)
    overlay = colormap * 0.4 + img * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    grad_cam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grad_cam.png')
    plt.imsave(grad_cam_path, overlay / 255)
    plt.close()

    return grad_cam_path, prediction

# -------------------- FLASK ROUTES --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        try:
            grad_cam_path, prediction = generate_grad_cam(model, file_path)

            additional_info = "Analysis includes Grad-CAM visualization highlighting areas of interest in the X-ray."
            detailed_report = generate_detailed_report(prediction, additional_info)
            pdf_path = create_pdf_report(prediction, grad_cam_path, detailed_report)

            return redirect(url_for('results', 
                                  filename=secure_filename(file.filename),
                                  prediction=float(prediction[0])))
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500

    return render_template('index.html')

@app.route('/results')
def results():
    filename = request.args.get('filename', '')
    prediction = request.args.get('prediction', '0.5')
    confidence = float(prediction) * 100
    
    # Determine diagnosis
    if float(prediction) > 0.5:
        diagnosis = 'Pneumonia Detected'
        status_class = 'positive'
    else:
        diagnosis = 'No Pneumonia'
        status_class = 'negative'
    
    return render_template('results.html', 
                         filename=filename, 
                         prediction=prediction,
                         confidence=confidence,
                         diagnosis=diagnosis,
                         status_class=status_class)

@app.route('/download/<filename>')
def download_pdf(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

@app.route('/grad-cam/<filename>')
def grad_cam(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype='image/png')

# -------------------- MAIN --------------------
if __name__ == '__main__':
    app.run(debug=True)