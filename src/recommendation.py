# def get_recommendation(prediction, probability):
#     """
#     Generate health recommendations based on prediction.
#     """

#     recommendations = []

#     if prediction == 1:

#         risk = "High"

#         recommendations.append("Consult a cardiologist as soon as possible.")
#         recommendations.append("Monitor blood pressure regularly.")
#         recommendations.append("Reduce salt and saturated fat intake.")
#         recommendations.append("Exercise for at least 30 minutes daily.")
#         recommendations.append("Avoid smoking and alcohol.")
#         recommendations.append("Maintain healthy body weight.")
#         recommendations.append("Get cholesterol and blood sugar checked.")
#         recommendations.append("Practice stress management (Yoga/Meditation).")

#     else:

#         risk = "Low"

#         recommendations.append("Continue maintaining a healthy lifestyle.")
#         recommendations.append("Exercise regularly.")
#         recommendations.append("Eat a balanced diet rich in fruits and vegetables.")
#         recommendations.append("Avoid smoking.")
#         recommendations.append("Drink plenty of water.")
#         recommendations.append("Sleep at least 7-8 hours daily.")
#         recommendations.append("Go for annual health check-ups.")

#     return {

#         "risk_level": risk,

#         "probability": round(probability * 100, 2),

#         "recommendations": recommendations

#     }
def get_recommendation(probability):

    risk = probability * 100

    if risk < 10:
        return {
            "level": "🟢 Very Low Risk",
            "recommendations": [
                "Maintain a balanced diet.",
                "Exercise at least 30 minutes daily.",
                "Get a yearly health check-up.",
                "Maintain a healthy BMI."
            ]
        }

    elif risk < 20:
        return {
            "level": "🟢 Low Risk",
            "recommendations": [
                "Continue healthy lifestyle.",
                "Reduce junk food.",
                "Avoid smoking and alcohol.",
                "Monitor blood pressure yearly."
            ]
        }

    elif risk < 40:
        return {
            "level": "🟡 Moderate Risk",
            "recommendations": [
                "Exercise 45 minutes daily.",
                "Reduce cholesterol-rich food.",
                "Monitor blood pressure regularly.",
                "Consult a doctor if symptoms appear."
            ]
        }

    elif risk < 60:
        return {
            "level": "🟠 High Risk",
            "recommendations": [
                "Consult a cardiologist.",
                "Control blood pressure and diabetes.",
                "Reduce salt intake.",
                "Regular ECG and blood tests."
            ]
        }

    elif risk < 80:
        return {
            "level": "🔴 Very High Risk",
            "recommendations": [
                "Immediate medical consultation.",
                "Follow prescribed medication.",
                "Quit smoking immediately.",
                "Monitor blood sugar and cholesterol."
            ]
        }

    else:
        return {
            "level": "🚨 Critical Risk",
            "recommendations": [
                "Seek immediate medical attention.",
                "Follow emergency medical advice.",
                "Do not ignore chest pain or breathing difficulty.",
                "Regular follow-up with a cardiologist."
            ]
        }