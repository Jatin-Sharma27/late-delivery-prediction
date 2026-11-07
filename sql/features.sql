-- One row per delivered order, using only information available at purchase.
-- Deliberately excluded: approval, carrier and delivery dates, and reviews.
WITH geo AS (
    SELECT geolocation_zip_code_prefix AS zip,
           AVG(geolocation_lat) AS lat,
           AVG(geolocation_lng) AS lng
    FROM geolocation
    WHERE geolocation_lat BETWEEN -34 AND 6      -- drop points outside Brazil
      AND geolocation_lng BETWEEN -74 AND -34
    GROUP BY geolocation_zip_code_prefix
),
items AS (
    SELECT i.order_id,
           COUNT(*) AS n_items,
           COUNT(DISTINCT i.seller_id) AS n_sellers,
           SUM(i.price) AS total_price,
           SUM(i.freight_value) AS total_freight,
           SUM(p.product_weight_g) AS total_weight_g,
           SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS total_volume_cm3
    FROM order_items i
    LEFT JOIN products p USING (product_id)
    GROUP BY i.order_id
),
primary_item AS (
    -- The most expensive item decides the seller and category for the order.
    SELECT order_id, seller_id, product_id
    FROM (
        SELECT order_id, seller_id, product_id,
               ROW_NUMBER() OVER (PARTITION BY order_id
                                  ORDER BY price DESC, order_item_id) AS rn
        FROM order_items
    )
    WHERE rn = 1
),
payment AS (
    SELECT order_id, payment_type, payment_installments
    FROM (
        SELECT order_id, payment_type, payment_installments,
               ROW_NUMBER() OVER (PARTITION BY order_id
                                  ORDER BY payment_value DESC, payment_sequential) AS rn
        FROM order_payments
    )
    WHERE rn = 1
)
SELECT o.order_id,
       o.order_purchase_timestamp AS purchase_ts,
       o.order_delivered_customer_date AS delivered_ts,
       o.order_estimated_delivery_date AS estimated_ts,
       CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
            THEN 1 ELSE 0 END AS is_late,
       julianday(date(o.order_estimated_delivery_date)) - julianday(o.order_purchase_timestamp) AS est_days,
       julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp) AS actual_days,
       it.n_items, it.n_sellers, it.total_price, it.total_freight,
       it.total_weight_g, it.total_volume_cm3,
       pi.seller_id,
       c.customer_state, s.seller_state,
       COALESCE(t.product_category_name_english, 'unknown') AS category,
       COALESCE(pay.payment_type, 'unknown') AS payment_type,
       pay.payment_installments,
       cg.lat AS customer_lat, cg.lng AS customer_lng,
       sg.lat AS seller_lat, sg.lng AS seller_lng
FROM orders o
JOIN items it ON it.order_id = o.order_id
JOIN primary_item pi ON pi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN sellers s ON s.seller_id = pi.seller_id
LEFT JOIN products pr ON pr.product_id = pi.product_id
LEFT JOIN category_translation t ON t.product_category_name = pr.product_category_name
LEFT JOIN payment pay ON pay.order_id = o.order_id
LEFT JOIN geo cg ON cg.zip = c.customer_zip_code_prefix
LEFT JOIN geo sg ON sg.zip = s.seller_zip_code_prefix
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
ORDER BY o.order_purchase_timestamp;
