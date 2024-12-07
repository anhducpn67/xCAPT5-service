/**** html/glass/js/home.js - Common scripts for the sidebar **********
 **
 **  Copyright © 2014 Regents of the University of Michigan
 **
 **  All rights reserved.                                            */

/*** Sticky sidebar **************************************************/

function getScrollTop()
{
    if (typeof window.pageYOffset !== 'undefined' ) {
        return window.pageYOffset;
    }

    var doc = document.documentElement;
    if (doc.clientHeight) {
        return doc.scrollTop;
    }

    return document.body.scrollTop;
}

function getViewportHeight()
{
    if (window.innerWidth != undefined) {
        return window.innerHeight;
    } else if (document.documentElement != undefined &&
            document.documentElement.clientWidth != undefined &&
            document.documentElement.clientWidth != 0) {
        return document.documentElement.clientHeight;
    } else {
        return document.getElementsByTagName('body')[0].clientHeight;
    }
}

function scrollSidebar()
{
    var scroll = getScrollTop();
    var bar = document.getElementById('z_sidebar');
    var barHeight = bar.offsetHeight;
    var barTop = bar.getBoundingClientRect()["top"] + scroll;
    var viewHeight = getViewportHeight();
    var pageHeight = document.documentElement.offsetHeight;
    var bodyHeight = document.getElementById('z_body').offsetHeight;

    if (bodyHeight == barHeight + 15) return;

    var offset;

    if (barHeight < viewHeight || scroll < barTop) {
        offset = scroll - 132;
    } else if (scroll + viewHeight > barTop + barHeight) {
        offset = scroll + viewHeight - barHeight - 160;
    }

    if (offset == undefined) return;
    if (offset < 13) {
        offset = 13;
    } else if (offset > bodyHeight - barHeight - 2) {
        offset = bodyHeight - barHeight - 2;
    }
    bar.style.marginTop = offset + "px";
}

/*** Properties ******************************************************/

window.onscroll = scrollSidebar;
window.onresize = scrollSidebar;
