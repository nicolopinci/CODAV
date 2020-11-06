//document.addEventListener('mousedown', removeDrags);
//document.addEventListener('dragstart', removeDrags);
//document.addEventListener('drag', removeDrags);
//document.addEventListener('dragend', removeDrags);
//document.addEventListener('mousemove', addResizes);

const minimum_size = 100;
let original_width = 0;
let original_height = 0;
let original_x = 0;
let original_y = 0;
let original_mouse_x = 0;
let original_mouse_y = 0;


function removeDrags(e) {
    let all_graph_divs = document.getElementsByClassName("graph_div");

    for (var i = 0; i < all_graph_divs.length; i++) {
        all_graph_divs.item(i).draggable = "false";
        all_graph_divs.item(i).addEventListener("dragstart", removeDrag);
        all_graph_divs.item(i).addEventListener("drag", removeDrag);
        all_graph_divs.item(i).addEventListener("dragend", removeDrag);
        all_graph_divs.item(i).addEventListener("mousedown", removeDrag);

    }
}

function removeDrag(e) {
    e.preventDefault();
    e.stopPropagation();
}


function addResizes() {
    let all_graph_divs = document.querySelectorAll(".resizeGraph");

    for (var i = 0; i < all_graph_divs.length; i++) {
        if (!all_graph_divs.item(i).className.includes("resizedAttached")) {
            all_graph_divs.item(i).draggable = "false";
            all_graph_divs.item(i).addEventListener("dragstart", addResize);
            all_graph_divs.item(i).addEventListener("drag", addResize);
            all_graph_divs.item(i).addEventListener("mousedown", addResize);
            all_graph_divs.item(i).className += " resizedAttached";
        }

    }
}

// Source: https://medium.com/the-z/making-a-resizable-div-in-js-is-not-easy-as-you-think-bda19a1bc53d
function addResize(e) {
    e.preventDefault();
    e.stopPropagation();
    original_width = parseFloat(getComputedStyle(this.parentNode, null).getPropertyValue('width').replace('px', ''));
    original_height = parseFloat(getComputedStyle(this.parentNode, null).getPropertyValue('height').replace('px', ''));
    original_x = this.parentNode.getBoundingClientRect().left;
    original_y = this.parentNode.getBoundingClientRect().top;
    original_mouse_x = e.pageX;
    original_mouse_y = e.pageY;
    this.addEventListener('mousemove', resize);
    this.addEventListener('drag', resize);
    this.addEventListener('mouseup', stopResize);
}


function resize(e) {

    const width = original_width + (e.pageX - original_mouse_x);
    const height = original_height + (e.pageY - original_mouse_y)
    if (width > minimum_size) {
        this.parentNode.style.width = width + 'px'
    }
    if (height > minimum_size) {
        this.parentNode.style.height = height + 'px'
    }
    
}


function stopResize() {
    this.removeEventListener('mousemove', resize)
}
 