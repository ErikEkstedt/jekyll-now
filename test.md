---
layout: post
title: ABC
extra_cool: extra
---


hej

<nav>
  {% for item in site.data.my_custom_list %}
    > <a href="{{ item.quote }}" {% if page.url == item.link %}class="current"{% endif %}>{{ item.name }}</a>
    <li>
    {{ item.note }}
    </li>
  {% endfor %}
</nav>

<article>
  <div>
    {{ page.extra_cool }}
  </div>
</article>


