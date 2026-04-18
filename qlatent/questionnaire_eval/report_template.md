# Questionnaire Dashboard

## Questionnaire Summary
**Name:** {{ questionnaire_name }}  
**Number of Items:** {{ num_items }}  
**Factors:**
{% for factor in factors %}
- {{ factor }}
{% endfor %}

---

## Silhouette Information
- **Average Silhouette:** {{ avg_silhouette }}
- **Standard Deviation:** {{ std_silhouette }}
- **Number of Negative Silhouettes:** {{ num_negative_silhouette }}

---

## Cronbach's Alpha Information
- **Total Alpha:** {{ cronbach_alpha_total }}  
**Factors:**
{% for factor, alpha in cronbach_alpha_factors.items() %}
- {{ factor }}: {{ alpha }}
{% endfor %}

---

## Model Information
- **Number of Models:** {{ num_models }}
- **Average Parameters per Model:** {{ avg_parameters }}
- **Standard Deviation of Parameters:** {{ std_parameters }}
