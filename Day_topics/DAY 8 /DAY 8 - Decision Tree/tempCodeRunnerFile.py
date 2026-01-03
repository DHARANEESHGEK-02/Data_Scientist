        pred_num = model.predict([[age, gender]])[0]

        prediction = interest_map_rev.get(pred_num, "Unknown")

    return render_template('index.html', prediction=prediction)