
//document.addEventListener("load", attachListeners);
//document.addEventListener("DOMContentLoaded", attachListeners);

function attachListeners() {
    try {
        document.getElementById("pred_button").addEventListener("click", show_predictions);
        document.getElementById("edu_button").addEventListener("click", show_education);
        document.getElementById("general_button").addEventListener("click", show_general);
    }
    catch {
        setTimeout(attachListeners, 500);
    }
}


function show_predictions() {
    document.getElementById("predictions_container").style.display = "block";
    document.getElementById("edu_container").style.display = "none";
    document.getElementById("analyses_container").style.display = "none";

    document.getElementById("pred_button").className = "currentlySelected";
    document.getElementById("edu_button").className = "";
    document.getElementById("general_button").className = "";

}


function show_education() {
    document.getElementById("predictions_container").style.display = "none";
    document.getElementById("edu_container").style.display = "block";
    document.getElementById("analyses_container").style.display = "none";

    document.getElementById("pred_button").className = "";
    document.getElementById("edu_button").className = "currentlySelected";
    document.getElementById("general_button").className = "";
}


function show_general() {
    document.getElementById("predictions_container").style.display = "none";
    document.getElementById("edu_container").style.display = "none";
    document.getElementById("analyses_container").style.display = "block";

    document.getElementById("pred_button").className = "";
    document.getElementById("edu_button").className = "";
    document.getElementById("general_button").className = "currentlySelected";
}