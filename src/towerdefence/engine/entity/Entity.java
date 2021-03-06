package towerdefence.engine.entity;

import towerdefence.engine.component.RenderComponent;
import towerdefence.engine.component.Component;
import java.util.ArrayList;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.GameplayState;

/**
 * Entity - A Game object that can interact with other objects and be
 * removed from the game dynamically
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Entity {

    String id;

    Vector2f position;
    float scale;
    float rotation;

    boolean isDead;

    ArrayList<RenderComponent> renderComponents = new ArrayList<RenderComponent>();
    RenderComponent singleRenderComponent;

    ArrayList<Component> components = null;
    
    private int type;
    private boolean delete;

    public Entity(String id)
    {
        this.id = id;

        components = new ArrayList<Component>();

        position = new Vector2f(0,0);
        scale = 1;
        rotation = 0;
    }

    public void AddComponent(Component component)
    {
        if(RenderComponent.class.isInstance(component)) {
            singleRenderComponent = (RenderComponent)component;
            renderComponents.add(singleRenderComponent);
        }

        component.setOwnerEntity(this);
        components.add(component);
    }

    public void RemoveComponent(Component component) {
        components.remove(component);
    }

    public Component getComponent(String id)
    {
        for(Component comp : components)
        {
            if( comp.getId().equalsIgnoreCase(id) ) {
                return comp;
            }
        }

        return null;
    }

    /*
     * Returns the exact pixel position of the sprite (only use for rendering)
     */
    public Vector2f getPosition()
    {
        return new Vector2f(position.x, position.y);
    }


    /* Given tilesize and x,y position, return tile position
     *
     * @param tilesize Size of tiles in pixels
     */
    public Vector2f getTilePosition()
    {
        return new Vector2f((int) Math.floor((position.x / GameplayState.TILESIZE)),
                (int) Math.floor((position.y / GameplayState.TILESIZE)));
    }

    public float getScale()
    {
        return scale;
    }

    public float getRotation()
    {
        return rotation;
    }

    public void killEntity() {
        isDead=true;
    }

    public boolean isDead() {
        return isDead;
    }

    public String getId()
    {
        return id;
    }

    public void setPosition(Vector2f position)
    {
        this.position = position;
    }

    public void setRotation(float rotate)
    {
        rotation = rotate;
    }

    public void setScale(float scale)
    {
        this.scale = scale;
    }

    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {
        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }
    }

    public void render(GameContainer gc, StateBasedGame sb, Graphics gr)
    {
        if(renderComponents != null) {
            for(RenderComponent sRenderComponent : renderComponents) {
            	sRenderComponent.render(gc, sb, gr);
            }
        }
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }
    
    public void deleteEntity() {
        delete=true;
    }

    /**
     * @return the delete
     */
    public boolean isDelete() {
        return delete;
    }


}
