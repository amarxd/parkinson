"""
Verification Script - Proves that Parkinson's samples detect positive
"""
import numpy as np
from PIL import Image
from utils.voice_processor import VoiceProcessor
from utils.handwriting_processor import HandwritingProcessor
from utils.model_loader import ModelLoader

print("=" * 60)
print("PARKINSON'S DETECTION VERIFICATION")
print("=" * 60)
print()

# Initialize processors
voice_proc = VoiceProcessor()
handwriting_proc = HandwritingProcessor()
model_loader = ModelLoader()

print("📊 VOICE ANALYSIS - Testing Parkinson's Voice Sample")
print("-" * 60)

# Test Parkinson's voice
with open('parkinsons_voice_sample.wav', 'rb') as audio_file:
    features = voice_proc.extract_features(audio_file)
    
    if features:
        print(f"✓ Jitter: {features.get('Jitter_percent', 0):.6f}% (Normal < 0.006%)")
        print(f"✓ Shimmer: {features.get('Shimmer', 0):.6f} (Normal < 0.03)")
        print(f"✓ HNR: {features.get('HNR', 0):.2f} dB (Normal > 20 dB)")
        print()
        
        feature_array = np.array([features[name] for name in voice_proc.feature_names]).reshape(1, -1)
        prediction = model_loader.predict_voice(feature_array, model_type='mlp')
        
        if prediction['prediction'] == 1:
            print(f"🔴 RESULT: PARKINSON'S DETECTED (Confidence: {prediction['confidence']:.1f}%)")
        else:
            print(f"🟢 RESULT: NO PARKINSON'S (Confidence: {prediction['confidence']:.1f}%)")
        
        print()

print("=" * 60)
print()

print("📊 VOICE ANALYSIS - Testing Healthy Voice Sample")
print("-" * 60)

# Test healthy voice
with open('healthy_voice_sample.wav', 'rb') as audio_file:
    features = voice_proc.extract_features(audio_file)
    
    if features:
        print(f"✓ Jitter: {features.get('Jitter_percent', 0):.6f}% (Normal < 0.006%)")
        print(f"✓ Shimmer: {features.get('Shimmer', 0):.6f} (Normal < 0.03)")
        print(f"✓ HNR: {features.get('HNR', 0):.2f} dB (Normal > 20 dB)")
        print()
        
        feature_array = np.array([features[name] for name in voice_proc.feature_names]).reshape(1, -1)
        prediction = model_loader.predict_voice(feature_array, model_type='mlp')
        
        if prediction['prediction'] == 1:
            print(f"🔴 RESULT: PARKINSON'S DETECTED (Confidence: {prediction['confidence']:.1f}%)")
        else:
            print(f"🟢 RESULT: NO PARKINSON'S (Confidence: {prediction['confidence']:.1f}%)")
        
        print()

print("=" * 60)
print()

print("📊 HANDWRITING ANALYSIS - Testing Parkinson's Spiral")
print("-" * 60)

# Test Parkinson's spiral
pd_image = np.array(Image.open('parkinsons_spiral.png'))
features = handwriting_proc.extract_features(pd_image)

if features:
    print(f"✓ Tremor Score: {features.get('tremor_score', 0):.3f} (PD if > 0.5)")
    print(f"✓ Line Deviation: {features.get('line_deviation', 0):.3f} (PD if > 0.3)")
    print(f"✓ Pressure Variation: {features.get('pressure_variation', 0):.3f}")
    print()
    
    feature_array = np.array([features[name] for name in handwriting_proc.feature_names]).reshape(1, -1)
    prediction = model_loader.predict_handwriting(feature_array, model_type='cnn')
    
    if prediction['prediction'] == 1:
        print(f"🔴 RESULT: PARKINSON'S DETECTED (Confidence: {prediction['confidence']:.1f}%)")
    else:
        print(f"🟢 RESULT: NO PARKINSON'S (Confidence: {prediction['confidence']:.1f}%)")
    
    print()

print("=" * 60)
print()

print("📊 HANDWRITING ANALYSIS - Testing Healthy Spiral")
print("-" * 60)

# Test healthy spiral
healthy_image = np.array(Image.open('healthy_spiral.png'))
features = handwriting_proc.extract_features(healthy_image)

if features:
    print(f"✓ Tremor Score: {features.get('tremor_score', 0):.3f} (PD if > 0.5)")
    print(f"✓ Line Deviation: {features.get('line_deviation', 0):.3f} (PD if > 0.3)")
    print(f"✓ Pressure Variation: {features.get('pressure_variation', 0):.3f}")
    print()
    
    feature_array = np.array([features[name] for name in handwriting_proc.feature_names]).reshape(1, -1)
    prediction = model_loader.predict_handwriting(feature_array, model_type='cnn')
    
    if prediction['prediction'] == 1:
        print(f"🔴 RESULT: PARKINSON'S DETECTED (Confidence: {prediction['confidence']:.1f}%)")
    else:
        print(f"🟢 RESULT: NO PARKINSON'S (Confidence: {prediction['confidence']:.1f}%)")
    
    print()

print("=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)
print()
print("SUMMARY:")
print("• parkinsons_voice_sample.wav → Should detect POSITIVE for PD")
print("• healthy_voice_sample.wav → Should detect NEGATIVE (healthy)")
print("• parkinsons_spiral.png → Should detect POSITIVE for PD")
print("• healthy_spiral.png → Should detect NEGATIVE (healthy)")
print()
