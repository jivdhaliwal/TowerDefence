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
public abstract class Entity {

    ArrayList<Component> components = null;

    protected boolean dead;
    private boolean delete;
    
    String id;

    Vector2f position;

    RenderComponent renderComponent = null;

    protected float rotation;
    private float scale;
    private int type;

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
            renderComponent = (RenderComponent)component;
        }

        component.setOwnerEntity(this);
        components.add(component);
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
    
    public void RemoveComponent(Component component) {
        components.remove(component);
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
        if(renderComponent != null) {
            renderComponent.render(gc, sb, gr);
        }
    }

    public String getId()
    {
        return id;
    }
    
    public void deleteEntity() {
        delete=true;
    }

    /*
     * Returns the exact pixel position of the sprite (only use for rendering)
     */
    public Vector2f getPosition()
    {
        return new Vector2f(position.x, position.y);
    }

    public float getRotation()
    {
        return rotation;
    }

    public float getScale()
    {
        return scale;
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

    public int getType() {
        return type;
    }

    public boolean isDead() {
        return dead;
    }

    /**
     * @return the delete
     */
    public boolean isDelete() {
        return delete;
    }

    public void killEntity() {
        dead=true;
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

    
    public void setType(int type) {
        this.type = type;
    }

    


}
