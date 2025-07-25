# RentMind AI for Landlords 🏠🤖

![GitHub repo size](https://img.shields.io/github/repo-size/AbdullahRaoo/RentMind-AI-for-Landlords?color=blue&style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/AbdullahRaoo/RentMind-AI-for-Landlords?color=yellow&style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/AbdullahRaoo/RentMind-AI-for-Landlords?color=red&style=flat-square)

## 🌟 Project Overview

![UI Preview](./UI.png)

RentMind AI is an intelligent platform designed to assist landlords in managing their properties efficiently. It leverages cutting-edge AI technologies to provide predictive maintenance, tenant screening, rent pricing, and more.

---

## 🛠️ Tech Stack

### **Backend**
- [![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/) High-level Python web framework.
- [![Channels](https://img.shields.io/badge/Channels-4.0-blue?style=for-the-badge)](https://channels.readthedocs.io/en/stable/) For real-time communication.
- [![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=for-the-badge)](https://faiss.ai/) Vector search for tenant screening.
- [![LangChain](https://img.shields.io/badge/LangChain-AI%20Infrastructure-green?style=for-the-badge)](https://www.langchain.com/) AI infrastructure for chatbot and decision-making.

### **Frontend**
- [![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/) React framework for server-side rendering.
- [![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/) Utility-first CSS framework.
- [![Radix UI](https://img.shields.io/badge/Radix%20UI-Accessible%20Components-blueviolet?style=for-the-badge)](https://www.radix-ui.com/) Accessible UI components.

### **AI/ML**
- [![OpenAI](https://img.shields.io/badge/OpenAI-GPT%20Integration-purple?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/) GPT-based chatbot integration.
- [![XGBoost](https://img.shields.io/badge/XGBoost-Rent%20Pricing-red?style=for-the-badge)](https://xgboost.readthedocs.io/) Rent pricing model.
- [![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-Tenant%20Screening-blue?style=for-the-badge)](https://www.sbert.net/) For tenant screening.

---

## 🚀 Features

- **Tenant Screening**: AI-powered tenant evaluation.
- **Predictive Maintenance**: Alerts for property maintenance.
- **Dynamic Rent Pricing**: AI-driven rent suggestions.
- **Chatbot Integration**: Real-time assistance for landlords.

---

## 📂 Project Structure

```plaintext
RentMind-AI-for-Landlords/
├── backend/                # Django backend
├── front/                  # Next.js frontend
├── AI Assistant/           # AI utilities and scripts
├── predictive_maintenance_ai/ # Predictive maintenance models
├── Rent Pricing AI/        # Rent pricing models
```

---

## 🧩 Pipelines

### **1. Tenant Screening**
- **Input**: Tenant data.
- **Process**: FAISS vector search + Sentence Transformers.
- **Output**: Tenant suitability score.

### **2. Predictive Maintenance**
- **Input**: Property data.
- **Process**: ML model for maintenance prediction.
- **Output**: Maintenance alerts.

### **3. Rent Pricing**
- **Input**: Property and market data.
- **Process**: XGBoost model.
- **Output**: Optimal rent price.

---

## 🛠️ Setup Instructions

### **Backend**
1. Clone the repository:
   ```bash
   git clone https://github.com/AbdullahRaoo/RentMind-AI-for-Landlords.git
   ```
2. Navigate to the backend folder:
   ```bash
   cd backend
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   python manage.py runserver
   ```

### **Frontend**
1. Navigate to the frontend folder:
   ```bash
   cd front
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

---

## 📈 Future Enhancements

- **Integration with IoT devices** for real-time property monitoring.
- **Advanced analytics** for landlord decision-making.
- **Mobile app** for on-the-go management.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🌐 Connect

[![GitHub](https://img.shields.io/badge/GitHub-AbdullahRaoo-blue?style=flat-square&logo=github)](https://github.com/AbdullahRaoo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-AbdullahRaoo-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/abdullahraoo/)
